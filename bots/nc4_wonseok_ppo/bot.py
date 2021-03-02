__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

# python -m bots.nc_example_v5.bot --server=172.20.41.105
# kill -9 $(ps ax | grep SC2_x64 | fgrep -v grep | awk '{ print $1 }')
# kill -9 $(ps ax | grep bots.nc_example_v5.bot | fgrep -v grep | awk '{ print $1 }')
# ps aux

import os

from skimage.metrics import normalized_root_mse

from sc2.unit import Unit

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pathlib
import pickle
import time
import math

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.data import CloakState
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint
from sc2.pixel_map import PixelMap
import random

import os

# .을 const 앞에 왜 찍는 거지?
from .consts import ArmyStrategy, CommandType, EconomyStrategy

INF = 1e9

nest_asyncio.apply()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # wonseok add #
        self.fc1 = nn.Linear(5 + (len(EconomyStrategy) * 2) + 2, 128)
        # wonseok end #
        self.norm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 128)
        self.norm3 = nn.LayerNorm(128)
        self.vf = nn.Linear(128, 1)
        self.economy_head = nn.Linear(128, len(EconomyStrategy))
        self.army_head = nn.Linear(128, len(ArmyStrategy))

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        value = self.vf(x)
        economy_logp = torch.log_softmax(self.economy_head(x), -1)
        army_logp = torch.log_softmax(self.army_head(x), -1)
        bz = x.shape[0]
        logp = (economy_logp.view(bz, -1, 1) + army_logp.view(bz, 1, -1)).view(bz, -1)
        return value, logp


class Bot(sc2.BotAI):
    """
    example v1과 유사하지만, 빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    """

    def __init__(self, step_interval=5.0, host_name='', sock=None, name=None, version=""):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        self.name = name
        ## donghyun edited ##
        if sock is None:
            try:
                self.model = Model()
                checkpoint = pathlib.Path(__file__).parent / ('model' + version + '.pt')
                
                # gpu
                #checkpoint = torch.load(checkpoint)
                #self.model.load_state_dict(checkpoint['model_state_dict'])
                #self.model.to(torch.device("cuda"))
                # cpu
                checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
                #self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.load_state_dict(checkpoint)
            except Exception as exc:
                import traceback;
                traceback.print_exc()
        ## donghyun end ##


    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.can_attack_air_units = [UnitTypeId.MARINE, UnitTypeId.GHOST, UnitTypeId.BATTLECRUISER, UnitTypeId.VIKINGFIGHTER, UnitTypeId.THOR, UnitTypeId.THORAP] 

        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval
        self.evoked = dict()
        self.enemy_exists = dict()

        # 현재 병영생산전략
        self.economy_strategy = EconomyStrategy.MARINE.value
        self.next_unit = None
        # 현재 군대전략
        self.army_strategy = ArmyStrategy.DEFENSE

        # offense mode?
        self.offense_mode = False
        # 핵 보유?
        self.has_nuke = False
        self.train_raven = False
        self.map_height = 63
        self.map_width = 128
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색
        self.enemy_cc_health_percentage = 1.0
        # 내 그룹 initiate
        self.my_groups = []
        # 적 그룹 initiate
        self.enemy_groups = []

        # (32.5, 31.5) or (95.5, 31.5)
        if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0:
            self.enemy_cc = Point2(Point2((95.5, 31.5)))  # 적 시작 위치
            # self.enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치
            self.ready_left = Point2(((self.cc.position.x + self.enemy_cc.x) / 4, (self.cc.position.y + self.enemy_cc.y) / 2))
            self.ready_center = Point2(((self.cc.position.x + self.enemy_cc.x) / 2, (self.cc.position.y + self.enemy_cc.y) / 2))
            self.ready_right = Point2(((self.cc.position.x + self.enemy_cc.x) / 4 * 3, (self.cc.position.y + self.enemy_cc.y) / 2))
            
        else:
            self.enemy_cc = Point2(Point2((32.5, 31.5)))  # 적 시작 위치
            self.ready_left = Point2(((self.cc.position.x + self.enemy_cc.x) / 4 * 3, (self.cc.position.y + self.enemy_cc.y) / 2))
            self.ready_center = Point2(((self.cc.position.x + self.enemy_cc.x) / 2, (self.cc.position.y + self.enemy_cc.y) / 2))
            self.ready_right = Point2(((self.cc.position.x + self.enemy_cc.x) / 4, (self.cc.position.y + self.enemy_cc.y) / 2))
            

        # 벌쳐 정찰을 위한 좌표
        self.left_down = Point2((12.5, 10.5)) if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0 else Point2((110.5, 10.5))
        self.right_down = Point2((110.5, 10.5)) if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0 else Point2((12.5, 10.5))
        self.left_up = Point2((12.5, 51.5)) if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0 else Point2((110.5, 51.5))
        self.right_up = Point2((110.5, 51.5)) if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0 else Point2((12.5, 51.5))

        self.origin_scout = None

        # 아군 로봇에게 수리받는 중인 애들 list cache
        self.being_repaired_units_list = []

        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        # data = (JOIN, game_id)
        # self.sock.send_multipart([pickle.dumps(d) for d in data])

    async def on_unit_destroyed(self, unit_tag):
        """ Override this in your bot class. """
        # 적, 아군 유닛 모두 해당이 되어야 할텐데.. (적은 확인)
        self.enemy_exists.pop(unit_tag, None)
        # 아군 정찰 화염차가 죽을 때 scout_unit_dead_time 갱신
        if self.evoked.get(("scout_unit_tag"), None) is not None and unit_tag == self.evoked.get(("scout_unit_tag")):
            self.evoked["scout_unit_dead_time"] = self.time

    async def on_step(self, iteration: int):
        """
        매 frame마다 실행되는 함수
        """
        actions = list()  # 이번 step에 실행할 액션 목록

        if self.time - self.last_step_time >= self.step_interval:
            self.economy_strategy, self.army_strategy = self.set_strategy()
            self.last_step_time = self.time

        # set info
        self.combat_units = self.units.exclude_type(
            [UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC, UnitTypeId.MULE]
        )
        self.wounded_units = self.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색

        # self.cached_known_enemy_units = self.known_enemy_units()
        # self.cached_known_enemy_structures = self.known_enemy_structures()
        
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first # 왠지는 모르겠는데 이걸 추가해야 실시간 tracking이 된다..
        self.enemy_cc_cache = self.known_enemy_units(UnitTypeId.COMMANDCENTER) # 적 커맨드도 실시간으로 tracking!
        if not self.enemy_cc_cache.empty and self.enemy_cc_cache.first.is_visible:
            self.enemy_cc_health_percentage = self.enemy_cc_cache.first.health_percentage

        # 공격 모드가 아닌 기타 모드일때
        # offense_mode가 될지 말지 정함
        # 하나라도 트리거가 된다면 모두 트리거가 된다.
        ground_attack = False
        air_attack = False

        for unit in self.units.not_structure:
            who_attack = self.select_mode(unit)

            if ground_attack and air_attack :
                break

            if who_attack == "All":      
                # 모든 유닛이 트리거 작동
                ground_attack = True
                air_attack = True

            elif who_attack == "Ground" :
                # 지상 공격 가능한 유닛만
                ground_attack = True

            elif who_attack == "Air" :
                # 공중 공격 가능한 유닛만
                air_attack = True
                
        
        if ground_attack and air_attack :
            self.offense_mode = True
            for unit in self.units.not_structure:
                    if self.evoked.get(("scout_unit_tag"), None) is not None and unit.tag == self.evoked.get(
                            ("scout_unit_tag")):
                        self.evoked[(unit.tag, "offense_mode")] = False
                    else:
                        self.evoked[(unit.tag, "offense_mode")] = True
        
        elif ground_attack :
            self.offense_mode = True
            for unit in self.units.not_structure.filter(lambda u : u.can_attack_ground or u.type_id in [UnitTypeId.RAVEN, UnitTypeId.MEDIVAC, UnitTypeId.VIKINGFIGHTER,\
                                                                                                        UnitTypeId.BATTLECRUISER]):
                    if self.evoked.get(("scout_unit_tag"), None) is not None and unit.tag == self.evoked.get(
                            ("scout_unit_tag")):
                        self.evoked[(unit.tag, "offense_mode")] = False
                    else:
                        self.evoked[(unit.tag, "offense_mode")] = True
        
        elif air_attack :
            self.offense_mode = True
            for unit in self.units.not_structure.filter(lambda u : u.can_attack_air or u.type_id in [UnitTypeId.RAVEN, UnitTypeId.MEDIVAC, UnitTypeId.VIKINGASSAULT,\
                                                                                                     UnitTypeId.BATTLECRUISER]):
                    self.evoked[(unit.tag, "offense_mode")] = True
        else :
            for unit in self.units.not_structure :
                self.evoked[(unit.tag, "offense_mode")] = False
                self.offense_mode = False


        # 정찰할 화염차 선택
        if self.units(UnitTypeId.HELLION).exists:
            scout_unit_tag = self.evoked.get(("scout_unit_tag"), -1)
            if scout_unit_tag == -1:
                self.evoked[("scout_unit_tag")] = self.units(UnitTypeId.HELLION).first.tag
                self.evoked[(self.units(UnitTypeId.HELLION).first.tag, "offense_mode")] = False
                scout_unit_tag = self.evoked.get(("scout_unit_tag"))
            else:
                scout_exist = False
                for unit in self.units(UnitTypeId.HELLION):
                    if scout_unit_tag == unit.tag:
                        scout_exist = True
                        break
                if not scout_exist:
                    self.evoked[("scout_unit_tag")] = self.units(UnitTypeId.HELLION).first.tag
                    self.evoked[(self.units(UnitTypeId.HELLION).first.tag, "offense_mode")] = False
                    scout_unit_tag = self.evoked.get(("scout_unit_tag"))

        # 상대 목록 갱신
        for unit in self.known_enemy_units.not_structure:
            if self.enemy_exists.get(unit.tag, None) is None:
                self.enemy_exists[unit.tag] = unit.type_id

        # 아군 그룹 정보 갱신
        if not self.units.not_structure.empty:
            self.my_groups = self.unit_groups()
        # 적 그룹 정보 갱신
        self.enemy_groups = self.enemy_unit_groups()

        # 아군 수리받는 메카닉 유닛들 cache
        self.being_repaired_units_list = []
        for unit in self.units.not_structure:
            if unit.type_id is UnitTypeId.MULE and unit.is_repairing and self.evoked.get((unit.tag, "being_repaired_unit"), None) is not None:
                self.being_repaired_units_list.append(self.evoked.get((unit.tag, "being_repaired_unit")))

        actions += await self.train_action()
        actions += await self.unit_actions()

        await self.do_actions(actions)

    def set_strategy(self):
        #
        # 특징 추출
        #

        # 일반 state 요소 개수
        # 이를 편집한다면 Model의 init 함수에서도 input size 변경 필수!
        remain_state_cnt = 5

        # 아군 핵 보유 상태를 기록 +1
        # 아군 offense_mode 기록 +1
        state = np.zeros(remain_state_cnt + (len(EconomyStrategy) * 2) + 2, dtype=np.float32)

        state[0] = self.cc.health_percentage
        # 적 커맨드 HP 상황
        # snapshot으로 남아 있을 때는 마지막으로 확인된 HP를 사용
        #state[1] = self.enemy_cc_health_percentage

        state[1] = max(1, self.minerals / 1000)
        state[2] = max(1, self.vespene / 1000)
        state[3] = max(1, self.time / 360)
        state[4] = self.state.score.total_damage_dealt_life / 2500

        # 아군 유닛 state
        for unit in self.units.not_structure:
            if unit.type_id is UnitTypeId.THORAP:
                state[remain_state_cnt + EconomyStrategy.to_index[EconomyStrategy.THOR.value]] += 1
            elif unit.type_id is UnitTypeId.VIKINGASSAULT:
                state[remain_state_cnt + EconomyStrategy.to_index[EconomyStrategy.VIKINGFIGHTER.value]] += 1
            elif unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                state[remain_state_cnt + EconomyStrategy.to_index[EconomyStrategy.SIEGETANK.value]] += 1
            else:
                state[remain_state_cnt + EconomyStrategy.to_index[unit.type_id]] += 1

        # wonseok add #
        # 적 유닛 state
        for type_id in self.enemy_exists.values():
            if type_id is UnitTypeId.THORAP:
                state[remain_state_cnt + len(EconomyStrategy) + EconomyStrategy.to_index[EconomyStrategy.THOR.value]] += 1
            elif type_id is UnitTypeId.VIKINGASSAULT:
                state[remain_state_cnt + len(EconomyStrategy) + EconomyStrategy.to_index[EconomyStrategy.VIKINGFIGHTER.value]] += 1
            elif type_id is UnitTypeId.SIEGETANKSIEGED:
                state[remain_state_cnt + len(EconomyStrategy) + EconomyStrategy.to_index[EconomyStrategy.SIEGETANK.value]] += 1
            else:
                state[remain_state_cnt + len(EconomyStrategy) + EconomyStrategy.to_index[type_id]] += 1

        # has nuke?
        state[remain_state_cnt + (len(EconomyStrategy) * 2)] = self.has_nuke

        # offense_mode
        state[remain_state_cnt + (len(EconomyStrategy) * 2) + 1] = self.offense_mode

        state = state.reshape(1, -1)
        # wonseok end #

        # NN
        data = [
            CommandType.STATE,
            pickle.dumps(self.game_id),
            pickle.dumps(state.shape),
            state,
        ]
        if self.sock is not None:
            self.sock.send_multipart(data)
            data = self.sock.recv_multipart()
            value = pickle.loads(data[0])
            action = pickle.loads(data[1])
        else:
            with torch.no_grad():
                value, logp = self.model(torch.FloatTensor(state))
                value = value.item()
                action = logp.exp().multinomial(num_samples=1).item()

        economy_strategy = EconomyStrategy.to_type_id[action // len(ArmyStrategy)]
        army_strategy = ArmyStrategy(action % len(ArmyStrategy))
        return economy_strategy, army_strategy

    async def train_action(self):
        #
        # 사령부 명령 생성
        #
        actions = list()

        update_flag = False

        if self.known_enemy_units.not_structure.exists and not (self.known_enemy_units.not_structure.amount == 1 and \
                                                               self.known_enemy_units.not_structure.first.type_id is UnitTypeId.MARINE):
            update_flag = True

        if update_flag or self.next_unit is None :
            if self.economy_strategy is UnitTypeId.RAVEN :
                return actions
            self.next_unit = self.economy_strategy

        # 밤까마귀 하드코딩
        self.train_raven = False
        if self.units(UnitTypeId.RAVEN).empty:
            # self.enemy_exists는 [unit.tag] 키를 가짐
            for eunit_type_id in self.enemy_exists.values():
                if eunit_type_id in [UnitTypeId.GHOST, UnitTypeId.BANSHEE]:
                    self.train_raven = True
                    break

        if self.train_raven:
            if self.vespene >= 175:
                self.next_unit = UnitTypeId.RAVEN
            elif self.next_unit in [UnitTypeId.MARAUDER, UnitTypeId.GHOST, UnitTypeId.BATTLECRUISER,
                                    UnitTypeId.SIEGETANK, UnitTypeId.REAPER, \
                                    UnitTypeId.THOR, UnitTypeId.VIKINGFIGHTER, UnitTypeId.BANSHEE, UnitTypeId.NUKE]:
                self.next_unit = None

        if self.next_unit is None:
            return actions

        # MULE 생산은 하드코딩으로 대체한다.
        # 커맨드 체력 정도에 따라 MULE이 원하는 숫자보다 적으면 생산
        desired_MULE_cnt = 0
        if self.cc.health_percentage <= 0.3:
            desired_MULE_cnt = 3
        elif self.cc.health_percentage <= 0.5:
            desired_MULE_cnt = 2
        elif self.cc.health_percentage <= 0.7:
            desired_MULE_cnt = 1

        our_MULE_cnt = self.units.filter(lambda u: u.type_id is UnitTypeId.MULE).amount

        if desired_MULE_cnt > 0 and our_MULE_cnt < desired_MULE_cnt:
            for i in range(desired_MULE_cnt - our_MULE_cnt):
                if await self.can_cast(self.cc, AbilityId.CALLDOWNMULE_CALLDOWNMULE, only_check_energy_and_cooldown=True):
                    if self.cc.position.x < 50:
                        mule_summon_point = Point2((self.cc.position.x - 5, self.cc.position.y))
                    else:
                        mule_summon_point = Point2((self.cc.position.x + 5, self.cc.position.y))
                        # 정해진 곳에 MULE 소환
                    actions.append(self.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_summon_point))
                else:
                    break

        # 핵
        if self.next_unit == EconomyStrategy.NUKE.value:
            if self.can_afford(AbilityId.BUILD_NUKE) and not self.has_nuke and self.time - self.evoked.get(
                    (self.cc.tag, 'train'), 0) > 1.0:
                actions.append(self.cc(AbilityId.BUILD_NUKE))
                self.has_nuke = True
                self.evoked[(self.cc.tag, 'train')] = self.time
        # 지게로봇
        # 학습에 맡기지 않고 하드코딩을 하기로 결정.
        elif self.next_unit == EconomyStrategy.MULE.value:
            pass
            # if await self.can_cast(self.cc, AbilityId.CALLDOWNMULE_CALLDOWNMULE, only_check_energy_and_cooldown=True):
            #     if self.cc.position.x < 50:
            #         mule_summon_point = Point2((self.cc.position.x - 5, self.cc.position.y))
            #     else:
            #         mule_summon_point = Point2((self.cc.position.x + 5, self.cc.position.y))
            #         # 정해진 곳에 MULE 소환
            #     actions.append(self.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_summon_point))
        # 나머지
        elif self.can_afford(self.next_unit):
            if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(self.cc.train(self.next_unit))
                self.next_unit = None
                self.evoked[(self.cc.tag, 'train')] = self.time

        return actions

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def select_threat(self, unit: Unit):
        # 자신에게 위협이 될 만한 상대 유닛들을 리턴
        # 자신이 배틀크루저일때 예외처리 필요.. 정보가 제대로 나오나?
        threats = []
        if unit.is_flying or unit.type_id is UnitTypeId.BATTLECRUISER:
            threats = self.known_enemy_units.filter(
                lambda u: u.can_attack_air and u.air_range + 3 >= unit.distance_to(u))
            for eunit in self.known_enemy_units:
                if eunit.type_id is UnitTypeId.BATTLECRUISER and 6 + 3 >= unit.distance_to(eunit):
                    threats.append(eunit)
        else:
            threats = self.known_enemy_units.filter(
                lambda u: u.can_attack_ground and u.ground_range + 3 >= unit.distance_to(u))
            for eunit in self.known_enemy_units:
                if eunit.type_id is UnitTypeId.BATTLECRUISER and 6 + 3 >= unit.distance_to(eunit):
                    threats.append(eunit)

        return threats

    def select_mode(self, unit: Unit):
        # 정찰중인 벌쳐는 제외
        if unit.tag == self.evoked.get(("scout_unit_tag")):
            #self.evoked[(unit.tag, "offense_mode")] = False
            return None

        # 밴시, 밤까마귀, 지게로봇 제외
        if unit.type_id is UnitTypeId.BANSHEE or unit.type_id is UnitTypeId.MULE or unit.type_id is UnitTypeId.RAVEN :
            #self.evoked[(unit.tag, "offense_mode")] = False
            return None
        # 방어모드일때 공격모드로 전환될지 트리거 세팅
        # 방어모드라면 False, 공격모드로 바뀐다면 True return

        ground_targets = self.known_enemy_units.filter(
                lambda u: unit.distance_to(u) <= max(unit.sight_range, unit.ground_range, unit.air_range) and u.is_visible
                    and not u.is_flying)
        air_targets = self.known_enemy_units.filter(
                lambda u: unit.distance_to(u) <= max(unit.sight_range, unit.ground_range, unit.air_range) and u.is_visible
                    and u.is_flying)

        if ground_targets.exists and air_targets.exists :
            return "All"
        elif ground_targets.exists :
            return "Ground"
        elif air_targets.exists :
            return "Air"
        else :
            return None

        '''
        # 바이킹은 폼 변환이 있어서 target filter를 다르게 걸어준다.
        if unit.type_id in [UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT]:
            nearby_targets = self.known_enemy_units.filter(
                lambda u: unit.distance_to(u) <= max(unit.sight_range, unit.ground_range,
                                                     unit.air_range) and u.is_visible)
        else:
            nearby_targets = self.known_enemy_units.filter(
                lambda u: unit.distance_to(u) <= max(unit.sight_range, unit.ground_range, unit.air_range) and u.is_visible
                    and ((u.is_flying and unit.can_attack_air) or (not u.is_flying and unit.can_attack_ground)))
        if nearby_targets.empty:
            return None
        else:
            if unit.can_attack_air and unit.can_attack_ground 
            return True
        '''

    # 발각이 되었는가?
    def is_detected(self, unit) :
        is_revealed = False
        enemy_ravens = self.known_enemy_units(UnitTypeId.RAVEN)
        for e_raven in enemy_ravens :
            if unit.distance_to(e_raven) <= e_raven.detect_range :
                is_revealed = True
                break
        return is_revealed

 # 방어형 무빙샷
    # 차이점 : 쿨다운이 설정한 값보다 낮더라도 위협이 근처에 있으면 타겟에 대한 공격 명령과 동시에 철수 명령을 내림
    # 기대 효과 : 때릴 수 있으면 때리고 그렇지 못하면 접근을 하지 않을 것임
    # 무빙샷
    def defense_moving_shot(self, actions, unit, cooldown, target_func, margin_health: float = 0, minimum: float = 0):

        # print("WEAPON COOLDOWN : ", unit.weapon_cooldown)
        threats = self.select_threat(unit)
        check_threats = threats.filter(lambda u : not u.is_flying)

        if unit.weapon_cooldown < cooldown:
            target = target_func(unit)
            if self.time - self.evoked.get((unit.tag, "COOLDOWN"), 0.0) >= minimum:
                actions.append(unit.attack(target))
                self.evoked[(unit.tag, "COOLDOWN")] = self.time

        if (unit.weapon_cooldown >= cooldown or not check_threats.empty) and \
                (margin_health == 0 or unit.health_percentage <= margin_health) and self.time - self.evoked.get(
                (unit.tag, "COOLDOWN"), 0.0) >= minimum:  # 무빙을 해야한다면
            maxrange = 0
            total_move_vector = Point2((0, 0))

            # 자신이 클라킹 상태가 아닐 때나 클라킹 상태이지만 발각됬을 때
            is_revealed = False
            enemy_ravens = self.known_enemy_units(UnitTypeId.RAVEN)
            for e_raven in enemy_ravens:
                if unit.distance_to(e_raven) <= e_raven.detect_range:
                    is_revealed = True
                    break

            if not unit.is_cloaked or is_revealed:
                if not unit.is_flying:
                    # 배틀크루저 예외처리.
                    # 배틀은 can_attack_air/ground와 무기 범위가 다 false, 0이다.
                    for eunit in threats:
                        if eunit.type_id is UnitTypeId.BATTLECRUISER:
                            maxrange = max(maxrange, 6)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                        else:
                            maxrange = max(maxrange, eunit.ground_range)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (eunit.ground_range + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                else:
                    for eunit in threats:
                        if eunit.type_id is UnitTypeId.BATTLECRUISER:
                            maxrange = max(maxrange, 6)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                        else:
                            maxrange = max(maxrange, eunit.air_range)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (eunit.air_range + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector

                if not threats.empty:
                    total_move_vector /= math.sqrt(total_move_vector.x ** 2 + total_move_vector.y ** 2)
                    total_move_vector *= maxrange
                    # 이동!
                    dest = Point2((self.clamp(unit.position.x + total_move_vector.x, 0, self.map_width),
                                   self.clamp(unit.position.y + total_move_vector.y, 0, self.map_height)))
                    #print(self.time, " : ", dest)
                    actions.append(unit.move(dest))

        return actions

    # 무빙샷
    def moving_shot(self, actions, unit, cooldown, target_func, margin_health: float = 0, minimum: float = 0):

        # print("WEAPON COOLDOWN : ", unit.weapon_cooldown)
        threats = self.select_threat(unit)

        if unit.weapon_cooldown < cooldown:
            target = target_func(unit)
            if self.time - self.evoked.get((unit.tag, "COOLDOWN"), 0.0) >= minimum:
                actions.append(unit.attack(target))
                self.evoked[(unit.tag, "COOLDOWN")] = self.time

        elif (margin_health == 0 or unit.health_percentage <= margin_health) and self.time - self.evoked.get(
                (unit.tag, "COOLDOWN"), 0.0) >= minimum:  # 무빙을 해야한다면
            maxrange = 0
            total_move_vector = Point2((0, 0))

            # 자신이 클라킹 상태가 아닐 때나 클라킹 상태이지만 발각됬을 때
            is_revealed = False
            enemy_ravens = self.known_enemy_units(UnitTypeId.RAVEN)
            for e_raven in enemy_ravens :
                if unit.distance_to(e_raven) <= e_raven.detect_range :
                    is_revealed = True
                    break

            if not unit.is_cloaked or is_revealed:
                if not unit.is_flying:
                    # 배틀크루저 예외처리.
                    # 배틀은 can_attack_air/ground와 무기 범위가 다 false, 0이다.
                    for eunit in threats:
                        if eunit.type_id is UnitTypeId.BATTLECRUISER:
                            maxrange = max(maxrange, 6)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                        else:
                            maxrange = max(maxrange, eunit.ground_range)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (eunit.ground_range + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                else:
                    for eunit in threats:
                        if eunit.type_id is UnitTypeId.BATTLECRUISER:
                            maxrange = max(maxrange, 6)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                        else:
                            maxrange = max(maxrange, eunit.air_range)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (eunit.air_range + 3 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector

                if not threats.empty:
                    total_move_vector /= math.sqrt(total_move_vector.x ** 2 + total_move_vector.y ** 2)
                    total_move_vector *= maxrange
                    # 이동!
                    dest = Point2((self.clamp(unit.position.x + total_move_vector.x, 0, self.map_width),
                                   self.clamp(unit.position.y + total_move_vector.y, 0, self.map_height)))
                    actions.append(unit.move(dest))

        return actions

    # 적 유닛 그룹 정하기
    # 전차 시즈모드 공격을 위한 method
    # 오로지 적 지상유닛에 한하여 만들기
    def enemy_unit_groups(self):
        groups = []
        center_candidates = self.known_enemy_units.not_structure.filter(lambda
                                                                            u: not u.is_flying and u.is_visible)
        for eunit in center_candidates:
            group = center_candidates.closer_than(5, eunit)
            groups.append(group)

        groups.sort(key=lambda g: g.amount, reverse=True)
        ret_groups = []

        # groups가 비는 경우는 적군 지상 유닛이 아예 없다는 것
        # 이 경우 빈 리스트 반환
        if not groups:
            return []

        # groups가 비지 않는 경우
        ret_groups.append(groups[0])
        selected_units = groups[0]

        group_num = int(self.known_enemy_units.not_structure.amount / 10.0)

        for i in range(0, group_num):
            groups.sort(key=lambda g: (g - selected_units).amount, reverse=True)
            # groups.sorted(lambda g : g.filter(lambda u : not (u in selected_units)), reverse=True)
            ret_groups.append(groups[0])
            selected_units = selected_units or groups[0]

        return ret_groups

    # 유닛 그룹 정하기
    # 시즈탱크 제외하고 산정.
    # 그룹당 중복 가능..
    def unit_groups(self):
        groups = []
        center_candidates = self.units.not_structure.filter(
            lambda u: u.type_id is not UnitTypeId.SIEGETANKSIEGED and u.type_id is not UnitTypeId.SIEGETANK and u.type_id is not UnitTypeId.VIKINGFIGHTER)
        for unit in center_candidates:
            group = center_candidates.closer_than(5, unit)
            groups.append(group)

        groups.sort(key=lambda g: g.amount, reverse=True)
        ret_groups = []

        # groups가 비는 경우는 위에서 제외한 유닛을 제외하고 유닛이 아예 없다는 것
        # 이 경우 아군 커맨드가 그룹 센터가 되게끔 반환
        if not groups:
            return [self.units.structure]

        # groups가 비지 않는 경우
        ret_groups.append(groups[0])
        selected_units = groups[0]

        group_num = int(self.units.not_structure.amount / 10.0)

        for i in range(0, group_num):
            groups.sort(key=lambda g: (g - selected_units).amount, reverse=True)
            # groups.sorted(lambda g : g.filter(lambda u : not (u in selected_units)), reverse=True)
            ret_groups.append(groups[0])
            selected_units = selected_units or groups[0]

        return ret_groups

    async def unit_actions(self):
        #
        # 유닛 명령 생성
        #
        actions = list()

        # loc = await self.find_placement(building=(self.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE)), near=self.cc.position)
        # print(loc)
        # actions.append(self.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, loc))

        #print(self.army_strategy)

        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택

            # 수리로봇이 수리 중일 땐 움직이지 않게 한다.
            # 적이 근처까지 오지 않는 이상 명령을 받지 않음.
            if unit.tag in [repaired_unit.tag for repaired_unit in self.being_repaired_units_list] and self.select_threat(unit).empty:
                continue

            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(self.enemy_cc) < unit.distance_to(enemy_unit):
                target = self.enemy_cc
            else:
                target = enemy_unit

            #self.army_strategy = self.next_army_strategy

            if (not unit.type_id in [UnitTypeId.MEDIVAC, UnitTypeId.RAVEN, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED, \
                    UnitTypeId.MULE, UnitTypeId.BANSHEE]) and not self.evoked.get((unit.tag, "offense_mode"),False):
                #print("?")
                if unit.type_id is UnitTypeId.HELLION and unit.tag == self.evoked.get(("scout_unit_tag")):
                    pass
                elif self.army_strategy is ArmyStrategy.DEFENSE:
                    # self.cc를 attack하면 진짜 아군 커맨드를 친다 ㅠㅠ
                    actions.append(unit.attack(self.cc.position))
                elif self.army_strategy is ArmyStrategy.READY_LEFT:
                    actions.append(unit.attack(self.ready_left))
                elif self.army_strategy is ArmyStrategy.READY_CENTER:
                    actions.append(unit.attack(self.ready_center))
                elif self.army_strategy is ArmyStrategy.READY_RIGHT:
                    actions.append(unit.attack(self.ready_right))

            # offense strategy가 아닐 때 offense mode가 아니고 근처에 위협이 있을 때는
            # 자기가 공격하지 못하는 유닛이 자길 공격할 수 있는 것이므로 회피 기동 넣기
            if self.army_strategy is not ArmyStrategy.OFFENSE and not self.evoked.get((unit.tag, "offense_mode"),False) and\
                not self.select_threat(unit).empty and not unit.tag == self.evoked.get(("scout_unit_tag")) :

                # 밴시는 여기에 들어오면 대기중 혹은 자기들끼리 공격하러 간 경우
                banshee_escape = False
                if unit.type_id == UnitTypeId.BANSHEE :
                    # 은신중이고 안들켰으면?
                    if unit.is_cloaked and not self.is_detected(unit) :
                        pass # 회피기동 ㄴㄴ
                    else :
                       # 은신이 아니거나 들킨 상황이면 회피기동
                       banshee_escape = True

                elif not unit.type_id == UnitTypeId.BANSHEE or banshee_escape :
                    threats = self.select_threat(unit)
                    maxrange = 0
                    total_move_vector = Point2((0, 0))
                    if not unit.is_flying:
                        # 배틀크루저 예외처리.
                        # 배틀은 can_attack_air/ground와 무기 범위가 다 false, 0이다.
                        for eunit in threats:
                            if eunit.type_id is UnitTypeId.BATTLECRUISER:
                                maxrange = max(maxrange, 6)
                                move_vector = unit.position - eunit.position
                                move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                                total_move_vector += move_vector
                            else:
                                maxrange = max(maxrange, eunit.ground_range)
                                move_vector = unit.position - eunit.position
                                move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                move_vector *= (eunit.ground_range + 3 - unit.distance_to(eunit)) * 1.5
                                total_move_vector += move_vector
                    else:
                        for eunit in threats:
                            if eunit.type_id is UnitTypeId.BATTLECRUISER:
                                maxrange = max(maxrange, 6)
                                move_vector = unit.position - eunit.position
                                move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                                total_move_vector += move_vector
                            else:
                                maxrange = max(maxrange, eunit.air_range)
                                move_vector = unit.position - eunit.position
                                move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                move_vector *= (eunit.air_range + 3 - unit.distance_to(eunit)) * 1.5
                                total_move_vector += move_vector

                    if not threats.empty:
                        total_move_vector /= math.sqrt(total_move_vector.x ** 2 + total_move_vector.y ** 2)
                        total_move_vector *= maxrange
                        # 이동!
                        dest = Point2((self.clamp(unit.position.x + total_move_vector.x, 0, self.map_width),
                                    self.clamp(unit.position.y + total_move_vector.y, 0, self.map_height)))
                        actions.append(unit.move(dest))


            ## 의료선과 밤까마귀 아니면 ...
            if unit.type_id is not (UnitTypeId.MEDIVAC and UnitTypeId.RAVEN):

                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if (self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"),
                                                                                      False)) and unit.distance_to(
                            target) < 8:
                        # 유닛과 목표의 거리가 8이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time

                ## wonseok add ##

                ## MARINE ##
                if unit.type_id is UnitTypeId.MARINE:
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        def target_func(unit):
                            enemies = self.known_enemy_units.filter(lambda e: e.is_visible)
                            if not enemies.empty:
                                # 보이는 적이 하나라도 있다면, 그 중에 HP가 가장 적은 애 집중타격
                                return enemies.sorted(lambda u: u.health)[0]
                                # return enemies.closest_to(unit)
                            return self.enemy_cc

                        if unit.has_buff(BuffId.STIMPACK):
                            actions = self.moving_shot(actions, unit, 1, target_func, 0.5)
                        else:
                            actions = self.moving_shot(actions, unit, 3, target_func, 0.5)

                ## BATTLECRUISER ##
                if unit.type_id is UnitTypeId.BATTLECRUISER:
                    # 왜인지 모르지만 이놈은 관련 정보가 빠져 있다. 특히 attack range.
                    # 일단 공중, 지상 모두 range는 6.0
                    # 할 수 없이 하드코딩을 해야 할듯 하다.
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):

                        def target_func(unit):

                            # 커맨드 스냅샷 포함
                            # 만약 위협이 근처에 존재한다면 위협 제거
                            # 위협이 없다면 가까운애 때리러 가거나 야마토 포 쏘러 감
                            threats = self.select_threat(unit)
                            for eunit in self.known_enemy_units:
                                if eunit.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(eunit):
                                    threats.append(eunit)

                            if threats.empty:
                                if self.known_enemy_units.empty:
                                    return self.enemy_cc
                                else:
                                    return self.known_enemy_units.closest_to(unit)  # 가까운애 때리기
                            else:
                                return threats.sorted(lambda u: u.health)[0]

                        def yamato_target_func(unit):
                            # 야마토 포 상대 지정
                            # 일정 범위 내 적들에 한해 적용
                            yamato_enemy_range = 15

                            # 근처 밤까마귀가 있다면 야마토 포 우선순위 변경
                            yamato_candidate_id = [UnitTypeId.THORAP, UnitTypeId.THOR, UnitTypeId.BATTLECRUISER, \
                                                   UnitTypeId.RAVEN, UnitTypeId.VIKINGFIGHTER, UnitTypeId.BANSHEE,
                                                   UnitTypeId.SIEGETANKSIEGED,
                                                   UnitTypeId.SIEGETANK,]

                            for eunit_id in yamato_candidate_id:
                                target_candidate = self.known_enemy_units.filter(
                                    lambda u: u.type_id is eunit_id and unit.distance_to(u) <= yamato_enemy_range)
                                target_candidate.sorted(lambda u: u.health, reverse=True)
                                if not target_candidate.empty:
                                    return target_candidate.first


                            return self.enemy_cc

                        # 토르, 밤까마귀, 배틀 같은 성가시거나 피통 많은 애들을 조지는 데 야마토 포 사용
                        # 얘네가 없으면 아껴 놓다가 커맨드에 사용.
                        cruiser_abilities = await self.get_available_abilities(unit)
                        if AbilityId.YAMATO_YAMATOGUN in cruiser_abilities:
                            yamato_target = yamato_target_func(unit)
                            if type(yamato_target) is not Unit:
                                # 커맨드 unit을 가리키게 변경
                                if not self.known_enemy_units(UnitTypeId.COMMANDCENTER).empty:
                                    yamato_target = self.known_enemy_units(UnitTypeId.COMMANDCENTER).first
                                else :
                                    actions = self.moving_shot(actions, unit, 1, target_func, 0.3)
                                    continue
                            if unit.distance_to(yamato_target) >= 12:
                                actions = self.moving_shot(actions, unit, 1, yamato_target_func, 0.3)
                            else:
                                actions.append(unit(AbilityId.YAMATO_YAMATOGUN, yamato_target))  # 야마토 박고
                        # 야마토를 쓸 수 없거나 대상이 없다면 주위 위협 제거나 가까운 애들 때리러 간다.
                        else:
                            actions = self.moving_shot(actions, unit, 1, target_func, 0.3)

                        # 차원이동
                        # 야마토 포가 가능하고, 적 커맨드가 visible하고, 그 주위에 확인되는 위협이 없다면 적 커맨드로 차원이동.
                        # 그 외에는, 자신의 HP가 일정 수준 이하(0.1)일 때 아군 커맨드 위치로 순간이동.
                        # 어그로 빼는 용도(순간이동 동안은 무적)
                        if AbilityId.EFFECT_TACTICALJUMP in cruiser_abilities:
                            enemy_cc_structure = self.known_enemy_structures.first if not self.known_enemy_structures.empty else None
                            if AbilityId.YAMATO_YAMATOGUN in cruiser_abilities and enemy_cc_structure is not None and enemy_cc_structure.is_visible:
                                threats = self.select_threat(unit)
                                # 위협이 없고, 순간이동해야 이득인 먼 거리일 때 순간이동.
                                if threats.empty and unit.distance_to(self.enemy_cc) > 60:
                                    actions.append(unit(AbilityId.EFFECT_TACTICALJUMP, self.enemy_cc))  # 적 커맨드로 순간이동
                            if unit.health_percentage <= 0.1:
                                actions.append(unit(AbilityId.EFFECT_TACTICALJUMP, self.cc.position))  # 내 커맨드로 순간이동

                        ## 야마토 240 / 토르2방 나머지1방

                        # 탱크 토르 배틀 밤까 + 바이킹

                        ## 토르 우선적 -> 마나 많은 밤까마귀 -> 배틀 -> 체력 높은애

                    ## 마이크로 액션으로 날먹 호로록 구현하기

                ## BATTLECRUISER END ##

                ## THOR ##

                if unit.type_id is UnitTypeId.THOR:
                    splash_range = 0.5
                    enemy_cruisers = self.known_enemy_units(UnitTypeId.BATTLECRUISER)
                    enemy_vikings = self.known_enemy_units(UnitTypeId.VIKINGFIGHTER)
                    thor_attack_battle = False

                    # 배틀이랑 바이킹 존재하면 천벌포로 교체
                    if (enemy_cruisers.exists or enemy_vikings.exists) and self.evoked.get((unit.tag, "CHANGE_WEAPON"),
                                                                                           10.0) > 2.0:
                        distance = (enemy_cruisers or enemy_vikings).closest_distance_to(unit)
                        if distance < 15:
                            thor_attack_battle = True
                            actions.append(unit(AbilityId.MORPH_THORHIGHIMPACTMODE))  # 250mm 천벌포로 교체

                            self.evoked[(unit.tag, "CHANGE_WEAPON")] = self.time

                    # 또는 경장갑이 적으면 천벌포로 교체
                    elif self.evoked.get((unit.tag, "CHANGE_WEAPON"), 10.0) > 2.0:
                        flying_enemies = self.known_enemy_units.filter(lambda unit: unit.is_flying)  # 공중 유닛
                        if (flying_enemies.amount > 0 and flying_enemies.filter(
                                lambda u: u.is_light).amount < flying_enemies.filter(lambda u: u.is_armored).amount):  # 경장갑이 적으면
                            actions.append(unit(AbilityId.MORPH_THORHIGHIMPACTMODE))  # 250mm 천벌포로 교체
                            self.evoked[(unit.tag, "CHANGE_WEAPON")] = self.time

                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"),
                                                                                     False):  # 재블린 미사일 모드
                        def target_func(unit):


                            target = None
                            flying_enemies = self.known_enemy_units.filter(lambda unit: unit.is_flying)  # 공중 유닛
                            best_score = 0
                            for eunit in flying_enemies:
                                light_count = 0 if eunit.is_armored else 1
                                heavy_count = 0 if eunit.is_light else 1
                                for other in flying_enemies:
                                    if eunit.tag != other.tag and eunit.distance_to(other) < splash_range:
                                        if other.is_light:
                                            light_count += 1
                                        elif other.is_armored:
                                            heavy_count += 1
                                score = light_count * 2 + heavy_count
                                # 거리가 가까우고 먼 것도 점수에 넣을까..
                                if score > best_score:
                                    best_score = score
                                    target = eunit
                            # 가장 점수가 높았던 놈을 공격
                            # target=none이면 지상 유닛 중 가까운 놈 공격
                            if target is None:
                                target = self.enemy_cc
                                min_dist = math.sqrt(self.map_height ** 2 + self.map_width ** 2) + 10
                                for eunit in self.known_enemy_units:
                                    if eunit.is_visible and eunit.distance_to(unit) < min_dist:
                                        target = eunit
                                        min_dist = eunit.distance_to(unit)
                            return target

                        enemy_inrange_units = self.known_enemy_units.in_attack_range_of(unit)
                        if enemy_inrange_units.filter(lambda e: e.is_flying).empty:
                            actions.append(unit.attack(target_func(unit)))
                        else:
                            actions = self.moving_shot(actions, unit, 5, target_func, 0.3, 0.8)

                if unit.type_id is UnitTypeId.THORAP:
                    enemy_cruisers = self.known_enemy_units(UnitTypeId.BATTLECRUISER)
                    enemy_vikings = self.known_enemy_units(UnitTypeId.VIKINGFIGHTER)
                    thor_attack_battle = False

                    if enemy_cruisers.exists or enemy_vikings.exists:
                        distance = (enemy_cruisers or enemy_vikings).closest_distance_to(unit)
                        if distance < 15:
                            thor_attack_battle = True

                    elif self.evoked.get((unit.tag, "CHANGE_WEAPON"), 10.0) > 2.0:
                        flying_enemies = self.known_enemy_units.filter(lambda unit: unit.is_flying)  # 공중 유닛
                        if (flying_enemies.amount >= 0 and flying_enemies.filter(
                                lambda u: u.is_light).amount > flying_enemies.filter(lambda u: u.is_armored).amount):  # 경장갑이 많으면
                            actions.append(unit(AbilityId.MORPH_THOREXPLOSIVEMODE))  # 재블린 모드로 교체
                            self.evoked[(unit.tag, "CHANGE_WEAPON")] = self.time

                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"),
                                                                                     False):  # 250mm 천벌포 모드
                        def target_func(unit):
                            enemy_inrange_units = self.known_enemy_units.in_attack_range_of(unit)
                            enemy_flying_heavy = enemy_inrange_units.filter(lambda u: u.is_armored)

                            if self.known_enemy_units.empty:
                                target = self.enemy_cc
                            elif enemy_flying_heavy.empty:
                                target = self.known_enemy_units.closest_to(unit)
                            else:
                                target = enemy_flying_heavy.sorted(lambda e: e.health)[0]
                            return target

                        enemy_inrange_units = self.known_enemy_units.in_attack_range_of(unit)
                        if enemy_inrange_units.filter(lambda e: e.is_flying).empty:
                            actions.append(unit.attack(target_func(unit)))
                        else:
                            actions = self.moving_shot(actions, unit, 5, target_func, 0.3, 0.8)

                # 토르가 공성전차에겐 약하다.. 이속이 느려서
                # 이것에 대한 코드도 추가예정

                ## THOR END ##

                ## SIEGE TANK ##

                # 시즈탱크
                if unit.type_id is UnitTypeId.SIEGETANK:

                    our_other_units = self.units.not_structure - {unit}

                    # 전략이 offense거나 offense mode가 켜졌을 때
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get(
                            (unit.tag, "offense_mode"), False):
                        # 타겟을 삼아 그 타겟이 사정거리 안에 들어올 때까지 이동
                        # 이동 후 시즈모드.
                        # 시즈모드 박기 전에 위협이 근처에 존재하면 무빙샷

                        targets = self.known_enemy_units.filter(
                            lambda u: not u.is_flying and u.is_visible and unit.distance_to(u) <= 10)
                        target = None
                        armored_targets = targets.filter(lambda u: u.is_armored)
                        if not armored_targets.empty:
                            target = armored_targets.closest_to(unit)
                        else:
                            if not self.known_enemy_units.empty and self.enemy_groups:
                                target = self.known_enemy_units.closest_to(self.enemy_groups[0].center)
                            else:
                                target = self.enemy_cc

                        threats = self.select_threat(unit)

                        if (threats.empty or (not threats.empty and not threats.filter(
                                lambda u: u.type_id is UnitTypeId.SIEGETANKSIEGED).empty)) and not our_other_units.empty:
                            if unit.distance_to(target) > 10:
                                actions.append(unit.attack(target))
                            else:
                                actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                                self.evoked[(unit.tag, "Last_sieged_mode_time")] = self.time
                        else:
                            if not threats.empty:
                                def target_func(unit):
                                    targets = self.known_enemy_units.filter(
                                        lambda u: not u.is_flying and u.is_visible)
                                    if targets.empty:
                                        return self.enemy_cc
                                    else:
                                        return targets.closest_to(unit)

                                actions = self.moving_shot(actions, unit, 3, target_func)
                            if our_other_units.empty:
                                actions.append(unit.move(self.cc))


                    # 전략이 offense가 아니고 offense mode도 아님
                    else:

                        rally_point = None
                        if self.army_strategy is ArmyStrategy.DEFENSE:
                            rally_point = self.cc.position
                        elif self.army_strategy is ArmyStrategy.READY_LEFT:
                            rally_point = self.ready_left
                        elif self.army_strategy is ArmyStrategy.READY_CENTER:
                            rally_point = self.ready_center
                        elif self.army_strategy is ArmyStrategy.READY_RIGHT:
                            rally_point = self.ready_right

                        desired_pos = self.evoked.get((unit.tag, "desired_pos"), None)
                        # if desired_pos is None or (desired_pos is not None and\
                        #         not await self.can_place(building=AbilityId.SIEGEMODE_SIEGEMODE, position=desired_pos)):
                        if desired_pos is None or self.evoked.get((unit.tag, "rally_point"),
                                                                  self.enemy_cc) != rally_point:
                            dist = random.randint(3, 4)
                            dist_x = random.randint(1, dist)
                            dist_y = math.sqrt(dist ** 2 - dist_x ** 2) if random.randint(0,
                                                                                          1) == 0 else -math.sqrt(
                                dist ** 2 - dist_x ** 2)
                            desire_add_vector = Point2(
                                (-dist_x, dist_y)) if self.cc.position.x < 50 else Point2((dist_x, dist_y))
                            desired_pos = rally_point + desire_add_vector
                            desired_pos = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                                  self.clamp(desired_pos.y, 0, self.map_height)))
                            self.evoked[(unit.tag, "desired_pos")] = desired_pos
                            self.evoked[(unit.tag, "rally_point")] = rally_point

                        threats = self.select_threat(unit)

                        if threats.empty or (not threats.empty and not threats.filter(lambda u: u.type_id is UnitTypeId.SIEGETANKSIEGED).empty):
                            if unit.distance_to(desired_pos) >= 3.0:
                                actions.append(unit.move(desired_pos))
                            else:
                                actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                                self.evoked[(unit.tag, "Last_sieged_mode_time")] = self.time
                        else:
                            def target_func(unit):
                                targets = self.known_enemy_units.filter(
                                    lambda u: not u.is_flying and u.is_visible)
                                if targets.empty:
                                    return self.enemy_cc
                                else:
                                    return targets.closest_to(unit)

                            actions = self.moving_shot(actions, unit, 3, target_func)

                # 시즈모드 시탱
                # 시즈모드일 때는 공격모드이거나 offense mode가 켜졌을 때에 관해 행동을 기술
                # 방어모드이고 offense mode가 아닐 때에는 따로 필요없음
                if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get(
                            (unit.tag, "offense_mode"), False):
                        targets = self.known_enemy_units.filter(
                            lambda u: not u.is_flying and u.is_visible and unit.distance_to(
                                u) <= unit.ground_range)
                        armored_targets = targets.not_structure.filter(lambda u: u.is_armored)
                        if not armored_targets.empty:
                            target = armored_targets.closest_to(unit)
                            actions.append(unit.attack(target))
                        else:
                            if not self.known_enemy_units.not_structure.empty and self.enemy_groups:
                                target = self.known_enemy_units.not_structure.closest_to(
                                    self.enemy_groups[0].center)
                            else:
                                # 적 커맨드가 보이면 커맨드 유닛 자체를 타겟으로, 아니면 좌표로..
                                target = self.known_enemy_units(
                                    UnitTypeId.COMMANDCENTER).first if self.is_visible(self.enemy_cc) \
                                    else self.enemy_cc
                            if target in targets:
                                actions.append(unit.attack(target))
                            elif self.time - self.evoked.get((unit.tag, "Last_sieged_mode_time"),
                                                             -10.0) >= 7.0:
                                actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))

                    else:
                        # 만약 현재 시즈박은 위치가 원래 있어야 할 위치와 다르면 시즈모드 풀기
                        desired_pos = self.evoked.get((unit.tag, "desired_pos"), None)
                        if desired_pos is None:
                            rally_point = None
                            if self.army_strategy is ArmyStrategy.DEFENSE:
                                rally_point = self.cc.position
                            elif self.army_strategy is ArmyStrategy.READY_LEFT:
                                rally_point = self.ready_left
                            elif self.army_strategy is ArmyStrategy.READY_CENTER:
                                rally_point = self.ready_center
                            elif self.army_strategy is ArmyStrategy.READY_RIGHT:
                                rally_point = self.ready_right

                            desired_pos = self.evoked.get((unit.tag, "desired_pos"), None)
                            # if desired_pos is None or (desired_pos is not None and\
                            #         not await self.can_place(building=AbilityId.SIEGEMODE_SIEGEMODE, position=desired_pos)):
                            if desired_pos is None or self.evoked.get((unit.tag, "rally_point"),
                                                                      self.enemy_cc) != rally_point:
                                dist = random.randint(3, 4)
                                dist_x = random.randint(1, dist)
                                dist_y = math.sqrt(dist ** 2 - dist_x ** 2) if random.randint(0,
                                                                                              1) == 0 else -math.sqrt(
                                    dist ** 2 - dist_x ** 2)
                                desire_add_vector = Point2(
                                    (-dist_x, dist_y)) if self.cc.position.x < 50 else Point2((dist_x, dist_y))
                                desired_pos = rally_point + desire_add_vector
                                desired_pos = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                                      self.clamp(desired_pos.y, 0, self.map_height)))
                                self.evoked[(unit.tag, "desired_pos")] = desired_pos
                                self.evoked[(unit.tag, "rally_point")] = rally_point

                        if unit.distance_to(desired_pos) >= 3.0:
                            actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))

                ## SIEGE TANK END ##

                ## 화염차

                # 화염차
                # 경장갑 위주로 때린다 / 정찰용
                # 원래는 일꾼제거용인데 그게 없으니 테테전 가정 하에
                # 경장갑만 잡는다.
                # 때리더라도 일직선으로 나가기에, 그 안에 최대한 많이 애들을 집어넣어 때리는게 중요
                # 근데 그걸 구현을 어떻게 하지 ㅋㅋㅋ
                # 근처에 없으면 그냥 가까이 있는 지상유닛 아무나 때린다.
                if unit.type_id is UnitTypeId.HELLION:
                    # 정찰용 화염차여도 공격 정책일 때는 공격해야 한다.
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):

                        def target_func(unit):
                            # 경장갑 우선 타겟팅
                            # 유닛 중 가장 가까운 애 때리기.
                            # 경장갑이 없으면 나머지 중 가장 가까운 놈 때리기
                            targets = self.known_enemy_units.not_structure.not_flying
                            if targets.empty:
                                return self.enemy_cc # point2

                            '''
                            light_targets = targets.filter(lambda u: u.is_light)
                            # 경장갑이 타겟 중에 없으니 나머지 중 가장 가까운 놈 때리기
                            if light_targets.empty:
                                return targets.sorted(lambda u: unit.distance_to(u))[0]
                            # 경장갑
                            else:
                                return light_targets.sorted(lambda u: unit.distance_to(u))[0]
                            '''
                            return targets.sorted(lambda u : unit.distance_to(u))[0]

                        actions = self.moving_shot(actions, unit, 10, target_func)
                    # 내가 정찰용 화염차라면?
                    elif unit.tag == self.evoked.get(("scout_unit_tag")) and self.time - self.evoked.get(
                        (unit.tag, "end_time"), -8.0) >= 8.0 and (self.evoked.get("scout_unit_dead_time", None) is None or \
                    (self.evoked.get("scout_unit_dead_time", None) is not None and self.time - self.evoked.get("scout_unit_dead_time") >= 8.0)):# 원래는 5초

                        if self.evoked.get((unit.tag, "scout_routine"), []) == []:
                            self.evoked[(unit.tag, "scout_routine")] = ["Center", "RightUp", "RightDown", "LeftDown",
                                                                        "LeftUp", "End"]

                        # 적, 아군 통틀어 어떤 유닛, 건물이라도 만나면 정찰 방향 수정
                        our_other_units = self.units - {unit}
                        if (unit.is_idle or our_other_units.closer_than(4, unit).exists or self.known_enemy_units.closer_than(unit.sight_range - 2, unit).exists) \
                                and self.time - self.evoked.get((unit.tag, "scout_time"), -3.0) >= 3.0:

                            if self.evoked.get((unit.tag, "scout_mode"), None) is None:
                                self.evoked[(unit.tag, "scout_mode")] = "retreat"
                            else:
                                if self.evoked.get((unit.tag, "scout_mode")) == "ongoing":
                                    self.evoked[(unit.tag, "scout_mode")] = "retreat"
                                else:
                                    self.evoked[(unit.tag, "scout_mode")] = "ongoing"

                            if self.evoked.get((unit.tag, "scout_mode")) == "ongoing":

                                next = self.evoked.get((unit.tag, "scout_routine"))[0]

                                if next == "Center":
                                    actions.append(unit.move(self.enemy_cc))
                                    self.evoked[(unit.tag, "scout_routine")] = self.evoked.get((unit.tag, "scout_routine"))[1:]

                                elif next == "RightUp":
                                    if abs(unit.position.y - 31.5) > 5.0:
                                        actions.append(unit.move(self.right_up))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get(
                                            (unit.tag, "scout_routine"))[1:]
                                    else:
                                        actions.append(unit.move(Point2((unit.position.x, self.right_up.y))))


                                elif next == "RightDown":
                                    if abs(unit.position.y - 31.5) > 5.0:
                                        actions.append(unit.move(self.right_down))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get(
                                            (unit.tag, "scout_routine"))[1:]
                                    else:
                                        actions.append(unit.move(Point2((unit.position.x, self.right_down.y))))

                                elif next == "LeftDown":
                                    if abs(unit.position.y - 31.5) > 5.0:
                                        actions.append(unit.move(self.left_down))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get(
                                            (unit.tag, "scout_routine"))[1:]
                                    else:
                                        actions.append(unit.move(Point2((unit.position.x, self.left_down.y))))

                                elif next == "LeftUp":
                                    if abs(unit.position.y - 31.5) > 5.0:
                                        actions.append(unit.move(self.left_up))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get(
                                            (unit.tag, "scout_routine"))[1:]
                                    else:
                                        actions.append(unit.move(Point2((unit.position.x, self.left_up.y))))

                                elif next == "End":
                                    if self.army_strategy == ArmyStrategy.DEFENSE:
                                        actions.append(unit.move(self.cc))
                                    elif self.army_strategy == ArmyStrategy.READY_LEFT:
                                        actions.append(unit.move(self.ready_left))
                                    elif self.army_strategy == ArmyStrategy.READY_CENTER:
                                        actions.append(unit.move(self.ready_center))
                                    elif self.army_strategy == ArmyStrategy.READY_RIGHT:
                                        actions.append(unit.move(self.ready_right))

                                    self.evoked[(unit.tag, "end_time")] = self.time
                                    self.evoked[(unit.tag, "scout_routine")] = []

                            else:
                                if self.army_strategy == ArmyStrategy.DEFENSE:
                                    actions.append(unit.move(self.cc))
                                elif self.army_strategy == ArmyStrategy.READY_LEFT:
                                    actions.append(unit.move(self.ready_left))
                                elif self.army_strategy == ArmyStrategy.READY_CENTER:
                                    actions.append(unit.move(self.ready_center))
                                elif self.army_strategy == ArmyStrategy.READY_RIGHT:
                                    actions.append(unit.move(self.ready_right))
                                break

                            self.evoked[(unit.tag, "scout_time")] = self.time


                # 바이킹 전투기 모드(공중)
                if unit.type_id is UnitTypeId.VIKINGFIGHTER:
                    # 무빙샷이 필요
                    # 한방이 쎈 유닛이다.
                    # 타겟을 정해야 한다.
                    # 우선순위 1. 적의 공중 유닛 중 가장 hp가 적은 놈을 치고 빠지기
                    # 우선순위 2. 1이 해당되는 놈들이 없다면(적의 공중 유닛이 없다면) 탱크 중 가장 hp 없는 놈 바로 아래에 내려서 공격
                    # 탱크가 없다면 주위 임의의 지상 유닛을 때린다.
                    # 우선순위 3. 1,2가 해당되지 않는다면 바로 커맨드로 가서 변환 후 때리기
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        # print(unit)
                        # 랜딩 flag initiate
                        if self.evoked.get((unit.tag, "prepare_landing"), None) is None:
                            self.evoked[(unit.tag, "prepare_landing")] = False
                        # 우선순위 1
                        # 눈에 보이는(visible) 공중 유닛 대상
                        first_targets = self.known_enemy_units.filter(lambda e: e.is_flying and e.can_be_attacked)
                        if not first_targets.empty:

                            self.evoked["Last_enemy_aircraft_time"] = self.time

                            def target_func(unit):

                                # target = first_targets.sorted(lambda e: e.health)[0]
                                target = first_targets.closest_to(unit)
                                return target

                            # 생존율을 높이기 위한 방어형 무빙샷
                            actions = self.defense_moving_shot(actions, unit, 1, target_func)

                        # 우선순위 2
                        elif self.time - self.evoked.get("Last_enemy_aircraft_time", 0.0) >= 1.0:
                            # 이 경우 바이킹 그룹 말고 다른 아군 유닛이 있는가가 굉장히 중요함!
                            # 있으면 통상 메뉴얼대로, 없으면 가급적이면 아군 커맨드로 대피.
                            # 적 공중 유닛이 1초 이상 보이지 않는 경우에만 해당
                            # 유닛 그룹 중앙에서 내려서 싸울 것.
                            ground_targets = self.known_enemy_units.filter(
                                lambda u: u.type_id is not UnitTypeId.BANSHEE and u.is_visible)
                            our_other_units = self.units.not_structure - self.units(
                                {UnitTypeId.VIKINGFIGHTER, UnitTypeId.VIKINGASSAULT})
                            # 일정 시간 이상(1초)적 공중 유닛이 보이지 않고 그룹 센터로부터 일정 범위 안(3)에 들어온다면 착륙
                            if not ground_targets.empty:
                                # 랜딩 준비 단계가 아니면 준비 단계로 만든다.
                                if not self.evoked.get((unit.tag, "prepare_landing")):
                                    self.evoked[(unit.tag, "prepare_landing")] = True

                                ## 랜딩 준비단계일때, 착륙지점이 None이거나 None이 아닌데 landing_loc 계산
                                landing_loc = self.evoked.get((unit.tag, "landing_loc"), None)
                                if self.evoked.get((unit.tag, "prepare_landing")) and \
                                        (landing_loc is None or (landing_loc is not None and \
                                                                 (not (9 <= ground_targets.closest_distance_to(
                                                                     landing_loc) <= 15) or \
                                                                  not await self.can_place(
                                                                      building=AbilityId.MORPH_VIKINGASSAULTMODE,
                                                                      position=landing_loc)))):
                                    dist = random.randint(8, 10)
                                    dist_x = random.randint(7, dist)
                                    dist_y = math.sqrt(dist ** 2 - dist_x ** 2) \
                                        if random.randint(0, 1) == 0 else -math.sqrt(dist ** 2 - dist_x ** 2)
                                    desire_add_vector = Point2(
                                        (-dist_x, dist_y)) if self.cc.position.x < 50 else Point2((dist_x, dist_y))
                                    desired_pos = self.my_groups[0].center + desire_add_vector
                                    landing_loc = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                                          self.clamp(desired_pos.y, 0, self.map_height)))
                                    self.evoked[(unit.tag, "landing_loc")] = landing_loc

                                if self.evoked.get((unit.tag, "prepare_landing")) and \
                                        9 <= ground_targets.closest_distance_to(landing_loc) <= 15:
                                    if unit.distance_to(landing_loc) < 5.0:
                                        actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))
                                    else:
                                        actions.append(unit.move(landing_loc))

                            else:
                                if our_other_units.empty:
                                    actions.append(unit.move(self.cc.position))
                                else:
                                    actions.append(unit.attack(self.enemy_cc))
                        # 기다리는 동안은 계속 튀어라
                        else:
                            threats = self.select_threat(unit)  # 위협이 있으면 ㅌㅌ
                            if not threats.empty:
                                maxrange = 0
                                total_move_vector = Point2((0, 0))
                                for eunit in threats:
                                    if eunit.type_id is UnitTypeId.BATTLECRUISER:
                                        maxrange = max(maxrange, 6)
                                        move_vector = unit.position - eunit.position
                                        move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                        move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                                        total_move_vector += move_vector
                                    else:
                                        maxrange = max(maxrange, eunit.air_range)
                                        move_vector = unit.position - eunit.position
                                        move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                        move_vector *= (eunit.air_range + 3 - unit.distance_to(eunit)) * 1.5
                                        total_move_vector += move_vector

                                    total_move_vector /= math.sqrt(total_move_vector.x ** 2 + total_move_vector.y ** 2)
                                    total_move_vector *= maxrange
                                    # 이동!
                                    dest = Point2((self.clamp(unit.position.x + total_move_vector.x, 0, self.map_width),
                                                   self.clamp(unit.position.y + total_move_vector.y, 0,
                                                              self.map_height)))
                                    actions.append(unit.move(dest))


                # 바이킹 전투 모드(지상)
                # 공격 우선순위 : 공중유닛 > 사정거리 내 탱크 > 지상유닛 > 적 커맨드
                if unit.type_id is UnitTypeId.VIKINGASSAULT:

                    enemy_air = self.known_enemy_units.filter(lambda e: e.is_flying and e.can_be_attacked)
                    # 커맨드 때리는 동안은 공중모드로 변하지 않도록 함.
                    # 보이는 애들에 한해 타깃 선정!
                    ground_enemy_units = self.known_enemy_units.filter(lambda u: not u.is_flying and u.can_be_attacked)

                    # 아래 코드는 모드 상관없이 작동
                    # 랜딩을 마쳤으므로 랜딩 준비 flag를 다시 False로 되돌림
                    if self.evoked.get((unit.tag, "prepare_landing")):
                        self.evoked[(unit.tag, "prepare_landing")] = False

                    # 아래 코드는 모드 상관없이 작동
                    # 적의 지상 유닛이나 커맨드가 보이지 않거나 적 공중유닛이 나타나면 전투기로 변환
                    # 변환 후 로직은 공중 모드에 적혀 있다.
                    if not enemy_air.empty or ground_enemy_units.empty:
                        actions.append(unit(AbilityId.MORPH_VIKINGFIGHTERMODE))

                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        # 1. 적 지상유닛 중 가장 가까운 기계부터 공격
                        # 기계가 없으면 기타 나머지 중 가까운 놈부터
                        # 2. 적 커맨드
                        def target_func(unit):
                            ground_units = self.known_enemy_units.not_flying.filter(
                                lambda e: e.is_visible)

                            if not ground_units.empty:
                                enemy_machines = self.known_enemy_units.filter(lambda u: u.is_visible and u.is_mechanical)
                                if enemy_machines.empty:
                                    return ground_units.closest_to(unit)
                                else:
                                    return enemy_machines.closest_to(unit)

                            return self.enemy_cc

                        if not ground_enemy_units.empty:
                            actions = self.moving_shot(actions, unit, 1, target_func)

                        else:
                            actions.append(unit(AbilityId.MORPH_VIKINGFIGHTERMODE))

                ## REAPER ##

                if unit.type_id is UnitTypeId.REAPER and self.army_strategy is ArmyStrategy.OFFENSE:
                    if unit.health_percentage <= .4:  # 40퍼 이하면 도망
                        actions.append(unit.move(self.start_location))
                        self.evoked[(unit.tag, "REAPER_RUNAWAY")] = True
                    if self.evoked.get((unit.tag, "REAPER_RUNAWAY"), False):
                        if unit.health_percentage >= 0.7:  # 70퍼 이상이면 다시 전투 진입
                            self.evoked[(unit.tag, "REAPER_RUNAWAY")] = False
                    if not self.evoked.get((unit.tag, "REAPER_RUNAWAY"), False):
                        def target_func(unit):
                            ground_units = self.known_enemy_units.not_flying.filter(
                                lambda e: e.is_visible)
                            if not ground_units.empty:
                                return ground_units.closest_to(unit)
                            else:
                                return self.enemy_cc
                        actions = self.moving_shot(actions, unit, 1, target_func, 0.5)
                        # 가까운 적과 거리 비례로 도망가기

                ## REAPER END

            ## MEDIVAC ##

            if unit.type_id is UnitTypeId.MEDIVAC:
                if self.wounded_units.exists:
                    wounded_unit = self.wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    if self.combat_units.exists:
                        actions.append(unit.move(self.my_groups[0].center))

            ## MEDIVAC end ##

            ### wonseok add ###

            ## RAVEN ##

            if unit.type_id is UnitTypeId.RAVEN:

                threats = self.select_threat(unit)  # 위협이 있으면 ㅌㅌ
                banshees = self.known_enemy_units(UnitTypeId.BANSHEE).closer_than(unit.sight_range, unit)
                our_auto_turrets = self.units(UnitTypeId.AUTOTURRET)

                if not (self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"),
                                                                                      False)) \
                        and banshees.exists and unit.energy > 50 and threats.empty:
                    if our_auto_turrets.empty or (
                            not our_auto_turrets.empty and our_auto_turrets.closest_distance_to(unit) < 10):
                        build_loc = banshees.center
                        if await self.can_place(building=AbilityId.BUILDAUTOTURRET_AUTOTURRET,
                                                position=build_loc):
                            actions.append(unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, build_loc))

                elif unit.distance_to(target) < 15 and unit.energy > 75 and \
                        (self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"),
                                                                                       False)):  # 적들이 근처에 있고 마나도 있으면
                    known_only_enemy_units = self.known_enemy_units.not_structure
                    if known_only_enemy_units.exists:  # 보이는 적이 있다면
                        enemy_amount = known_only_enemy_units.amount
                        not_antiarmor_enemy = known_only_enemy_units.filter(
                            # anitarmor가 아니면서 가능대상에서 1초가 지났을 때...
                            lambda unit: self.time - self.evoked.get((unit.tag, "ANTIARMOR_POSSIBLE"),
                                                                     0) >= 1.0
                            # (not unit.has_buff(BuffId.RAVENSHREDDERMISSILEARMORREDUCTION)) and
                        )
                        not_antiarmor_enemy_amount = not_antiarmor_enemy.amount

                        if not_antiarmor_enemy_amount / enemy_amount > 0.5:  # 안티아머 걸리지 않은게 절반 이상이면 미사일 쏘기 center에
                            enemy_center = not_antiarmor_enemy.center
                            select_unit = not_antiarmor_enemy.closest_to(enemy_center)
                            possible_units = known_only_enemy_units.closer_than(3,
                                                                                select_unit)  # 안티아머 걸릴 수 있는 애들

                            for punit in possible_units:
                                self.evoked[(punit.tag, "ANTIARMOR_POSSIBLE")] = self.time
                            self.evoked[(punit.tag, "ANTIARMOR_POSSIBLE")] = self.time

                            actions.append(unit(AbilityId.EFFECT_ANTIARMORMISSILE, select_unit))

                        else:  # 안티아머가 있으면 매트릭스 걸 놈 추적
                            for enemy in known_only_enemy_units:
                                if enemy.is_mechanical and enemy.has_buff(
                                        BuffId.RAVENSCRAMBLERMISSILE):  # 기계이고 락다운 안걸려있으면 (robotic은 로봇)
                                    actions.append(unit(AbilityId.EFFECT_INTERFERENCEMATRIX, enemy))
                    # 터렛 설치가 효과적일까 모르겠네 돌려보고 해보기
                else:
                    if not threats.empty:
                        maxrange = 0
                        total_move_vector = Point2((0, 0))
                        for eunit in threats:
                            if eunit.type_id is UnitTypeId.BATTLECRUISER:
                                maxrange = max(maxrange, 6)
                                move_vector = unit.position - eunit.position
                                move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                move_vector *= (6 + 3 - unit.distance_to(eunit)) * 1.5
                                total_move_vector += move_vector
                            else:
                                maxrange = max(maxrange, eunit.air_range)
                                move_vector = unit.position - eunit.position
                                move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                                move_vector *= (eunit.air_range + 3 - unit.distance_to(eunit)) * 1.5
                                total_move_vector += move_vector

                            total_move_vector /= math.sqrt(
                                total_move_vector.x ** 2 + total_move_vector.y ** 2)
                            total_move_vector *= maxrange
                            # 이동!
                            dest = Point2(
                                (self.clamp(unit.position.x + total_move_vector.x, 0, self.map_width),
                                 self.clamp(unit.position.y + total_move_vector.y, 0,
                                            self.map_height)))
                            actions.append(unit.move(dest))

                    else:
                        enemy_banshee = self.known_enemy_units(UnitTypeId.BANSHEE)

                        if enemy_banshee.exists:
                            banshee_in_raven = enemy_banshee.closer_than(9.5, unit)
                            # print(banshee_in_raven)
                            if banshee_in_raven.amount / enemy_banshee.amount >= 0.5:
                                actions.append(unit.move(self.cc))
                            else:
                                # print("???")
                                actions.append(unit.move(enemy_banshee.closest_to(unit).position))
                        elif self.units.not_structure.exists:  # 전투그룹 중앙 대기
                            actions.append(unit.move(self.my_groups[0].center))

            ## RAVEN END ##

            ## GHOST ##

            if unit.type_id is UnitTypeId.GHOST:

                # 아래 내용은 공격 정책이거나 offense mode가 아닐 시에도 항시 적용됨
                threats = self.select_threat(unit)

                # 근처에 위협이 존재할 시 클라킹
                if not threats.empty and not unit.has_buff(BuffId.GHOSTCLOAK) and unit.energy_percentage >= 0.3:
                    actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_GHOST))

                # 만약 주위에 아무도 자길 때릴 수 없으면 클락을 풀어 마나보충
                if not threats.empty:
                    self.evoked[(unit.tag, "GHOST_CLOAK")] = self.time

                if threats.empty and self.time - self.evoked.get((unit.tag, "GHOST_CLOAK"), 0.0) >= 5 \
                        and unit.has_buff(BuffId.GHOSTCLOAK):
                    actions.append(unit(AbilityId.BEHAVIOR_CLOAKOFF_GHOST))

                # 핵 관련 상태 변경
                if unit.is_using_ability(AbilityId.TACNUKESTRIKE_NUKECALLDOWN) and self.has_nuke:
                    self.has_nuke = False

                if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                    ghost_abilities = await self.get_available_abilities(unit)
                    if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities:  # 핵 보유 중이라면
                        if ((not unit.has_buff(BuffId.GHOSTCLOAK)) and unit.energy >= 75.0) or ((unit.has_buff(
                                BuffId.GHOSTCLOAK)) and unit.energy >= 20.0):  # 마나가 75 이상이라면 클로킹 쓰고 핵 쏘러 가자
                            actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                            visible_enemies = self.known_enemy_units.not_structure.filter(lambda e: e.is_visible)
                            if visible_enemies.amount >= 10:  # 보이는게 10마리 정도 이상이면 여기다가 쏘고

                                ## 군집 확인 하기 ##
                                enemy_center = visible_enemies.center
                                select_unit = visible_enemies.closest_to(enemy_center)  # 허공에 쏘는걸 방지하기 위해
                                actions.append(unit(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, select_unit.position))
                            else:  # 상대 커멘드로 쏘러가자
                                actions.append(unit(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, self.enemy_cc.position))
                        else:
                            actions.append(unit.attack(self.cc.position))  # 핵은 있는데 마나 없으면 마나 채우기
                    else:  # 핵 없으면 생체유닛한테 저격하기 / 공격 OK
                        def target_func(unit):


                            enemies = self.known_enemy_units
                            light_enemies = enemies.filter(lambda e: e.is_light)

                            if not light_enemies.empty:
                                target = light_enemies.closest_to(unit)
                            elif not enemies.empty:
                                target = enemies.closest_to(unit)
                            else:
                                target = self.enemy_cc

                            return target

                        if unit.energy >= 50.0:
                            # 저격이 아깝지 않은 애는 불곰밖에 없다.(생체 유닛 중)
                            enemy_MARAUDER = self.known_enemy_units.filter(lambda e: e.type_id is UnitTypeId.MARAUDER)
                            if not enemy_MARAUDER.empty:
                                target = enemy_MARAUDER.closest_to(unit)  # 가장 가까운 생체유닛 저격
                                actions.append(unit(AbilityId.EFFECT_GHOSTSNIPE, target))
                            else:
                                actions = self.moving_shot(actions, unit, 10, target_func)
                        else:
                            actions = self.moving_shot(actions, unit, 10, target_func)

            # 불곰
            # 지상 공격만 가능
            # 해병과 기동력은 비슷하나 중장갑에 쎄다.
            # 중장갑 위주로 짤라먹는 플레이를 하려면.. 본진 업그레이드로 스팀팩을 업그레이드 해 주어야 한다. 되어 있는 건가?
            # 위 코드를 보면 되어 있는 것 같기두.. 스팀팩 쓰고 안쓰고의 여부는 위에 이미 구현되어 있음
            # 우선순위 1. 공성전차를 의료선에 실어서 잡는 방식으로 운용 - 최소 5명 이상(아니면 학습에 맡길까?)
            # 2. 토르 잡는 데 사용(기동성으로 치고 빠지기).
            # 3. 의료선으로 실어서 가거나 무리지어서 몰려가서 커맨드 부시기
            # 근데 의료선에 타 있다가 내리는 걸 어떻게 detect하지..
            if unit.type_id is UnitTypeId.MARAUDER:
                if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                    # 목표물을 찾아, 치고 빠지기 구현
                    def target_func(unit):
                        # 우선순위 4가지
                        # 1. 시즈탱크
                        query_units = self.known_enemy_units.filter(lambda
                                                                               u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED).sorted(
                            lambda u: u.health + unit.distance_to(u))
                        if not query_units.empty:
                            return query_units.first
                        # 2. 토르
                        query_units = self.known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.THOR or u.type_id is UnitTypeId.THORAP).sorted(
                            lambda u: u.health + unit.distance_to(u))
                        if not query_units.empty:
                            return query_units.first
                        # 3. 지상 유닛 중 가까운거
                        query_units = self.known_enemy_units.not_structure.filter(lambda e: e.is_visible and not e.is_flying)
                        if not query_units.empty:
                            return query_units.closest_to(unit)
                        # 4. 커맨드
                        # 이때는 None 리턴
                        return None

                    if target_func(unit) is None:
                        actions.append(unit.attack(self.enemy_cc))
                    else:
                        if unit.has_buff(BuffId.STIMPACK):
                            actions = self.moving_shot(actions, unit, 3, target_func)
                        else:
                            actions = self.moving_shot(actions, unit, 6, target_func)


            # 밴시
            # 공성전차를 위주로 잡게 한다.
            # 기본적으로 공성전차를 찾아다니되 들키면 튄다 ㅎㅎ
            if unit.type_id is UnitTypeId.BANSHEE:

                # 아래 내용은 공격 정책이거나 offense mode가 아닐 시에도 항시 적용됨
                threats = self.select_threat(unit)
                # clock_threats = self.known_enemy_units.filter(
                #     lambda u: u.type_id is UnitTypeId.RAVEN and unit.distance_to(u) <= u.sight_range)

                # 근처에 위협이 존재할 시 클라킹
                # 하지만 정찰 유닛(마린 1기)만 있을 시에는 클라킹을 하지 않는다.
                # 이 경우는 하는 것이 손해!
                if not threats.empty and not (threats.amount == 1 and threats.first.type_id == UnitTypeId.MARINE) and \
                        not unit.has_buff(BuffId.BANSHEECLOAK) and unit.energy_percentage >= 0.2:
                    actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_BANSHEE))

                # 만약 주위에 아무도 자길 때릴 수 없으면 클락을 풀어 마나보충
                if not threats.empty:
                    self.evoked[(unit.tag, "BANSHEE_CLOAK")] = self.time

                if threats.empty and self.time - self.evoked.get((unit.tag, "BANSHEE_CLOAK"), 0.0) >= 10 \
                        and unit.has_buff(BuffId.BANSHEECLOAK):
                    actions.append(unit(AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE))

                # 공격 정책이거나 offense mode가 트리거됬을 시
                if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                    #print("offense")
                    def target_func(unit):
                        enemy_tanks = self.known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                        if enemy_tanks.amount > 0:
                            target = enemy_tanks.closest_to(unit.position)
                            return target
                        # 만약 탱크가 없다면 HP가 가장 적으면서 가까운 아무 지상 유닛이나, 그것도 없다면 커맨드 직행
                        else:
                            targets = self.known_enemy_units.filter(lambda u: u.type_id is not UnitTypeId.COMMANDCENTER and not u.is_flying)
                            max_dist = math.sqrt(self.map_height**2 + self.map_width**2)
                            if not targets.empty:
                                target = targets.sorted(lambda u: u.health_percentage + unit.distance_to(u)/max_dist)[0]
                                return target
                            else:
                                return self.enemy_cc

                    # 공격
                    # 은신 상태이면서 밤까마귀가 감지하고 있지 않으면 그냥 공격
                    # 그 상태가 아니라면 무빙샷.
                    # TODO : 은신이 감지되고 있는지 확인이 불가함. CloakState 작동 불가
                    # 무빙샷 함수 안에 cloak에 대한 예외처리도 되어 있다.
                    actions = self.moving_shot(actions, unit, 5, target_func)

                else :
                    #print("not offense")
                    # 밤까마귀 봤으면 공격 ㄴㄴ
                    is_raven = False
                    is_can_air = False

                    for e in self.enemy_exists.values() :
                        if e in self.can_attack_air_units :
                            is_can_air = True
                            break

                    for e in self.enemy_exists.values() :
                        if e is UnitTypeId.RAVEN :
                            is_raven = True
                            #print("escape")
                            break

                        

                    if is_raven and is_can_air :
                        #print("!!!")
                        if self.army_strategy is ArmyStrategy.DEFENSE:
                            move_position = self.cc.position
                            # move와 attack 둘 중 뭐가 나을까..?
                            actions.append(unit.move(move_position))
                        elif self.army_strategy is ArmyStrategy.READY_LEFT:
                            actions.append(unit.move(self.ready_left))
                        
                        elif self.army_strategy is ArmyStrategy.READY_CENTER:
                            actions.append(unit.move(self.ready_center))

                        elif self.army_strategy is ArmyStrategy.READY_RIGHT:
                            actions.append(unit.move(self.ready_right))
                        
                        continue

                    def target_func(unit):
                        enemy_tanks = self.known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                        if enemy_tanks.amount > 0:
                            target = enemy_tanks.closest_to(unit.position)
                            return target
                        # 만약 탱크가 없다면 HP가 가장 적으면서 가까운 아무 지상 유닛이나, 그것도 없다면 커맨드 직행
                        else:
                            targets = self.known_enemy_units.filter(lambda u: u.type_id is not UnitTypeId.COMMANDCENTER and not u.is_flying)
                            max_dist = math.sqrt(self.map_height**2 + self.map_width**2)
                            if not targets.empty:
                                target = targets.sorted(lambda u: u.health_percentage + unit.distance_to(u)/max_dist)[0]
                                return target
                            else:
                                return self.enemy_cc
                    # 클럭 상태거나 클럭을 할 수 있으면 공격하러 가기
                    if (not unit.has_buff(BuffId.BANSHEECLOAK) and unit.energy_percentage >= 0.2) or (unit.has_buff(BuffId.BANSHEECLOAK) and unit.energy >= 5) :
                        actions = self.moving_shot(actions, unit, 5, target_func)
                    else :
                        if self.army_strategy is ArmyStrategy.DEFENSE:
                            move_position = self.cc.position
                            # move와 attack 둘 중 뭐가 나을까..?
                            actions.append(unit.move(move_position))
                        elif self.army_strategy is ArmyStrategy.READY_LEFT:
                            actions.append(unit.move(self.ready_left))
                        
                        elif self.army_strategy is ArmyStrategy.READY_CENTER:
                            actions.append(unit.move(self.ready_center))

                        elif self.army_strategy is ArmyStrategy.READY_RIGHT:
                            actions.append(unit.move(self.ready_right))

            # 지게로봇
            # 에너지 50을 사용하여 소환
            # 일정 시간 뒤에 파괴된다.
            # 용도 : 사령부 수리 or 메카닉 유닛 수리

            if unit.type_id is UnitTypeId.MULE:
                if self.cc.health < self.cc.health_max:
                    # 커맨드 수리
                    actions.append(unit(AbilityId.EFFECT_REPAIR_MULE, self.cc))
                    if unit.is_repairing:
                        self.evoked[(unit.tag, "being_repaired_unit")] = self.cc
                    else:
                        self.evoked[(unit.tag, "being_repaired_unit")] = None
                else:
                    # 근처 수리 가능한 메카닉 애들을 찾아 수리
                    # 아마 커맨드 근처에 있는 애들이 될 것임.
                    # 어차피 일정 시간 뒤 파괴되므로 HP가 가장 적은 애들을 찾는 것보다는 근처 애들이 나음
                    repair_candidate = self.units.not_structure.filter(lambda u: u.is_mechanical and u.health_percentage < 0.5)

                    repair_target = self.evoked.get((unit.tag, "being_repaired_unit"))
                    # repair_target = self.units.find_by_tag(repair_target_tag)
                    if unit.is_repairing and repair_target is not None:
                        repair_target = self.evoked.get((unit.tag, "being_repaired_unit"))
                        actions.append(unit(AbilityId.EFFECT_REPAIR_MULE, repair_target))
                        self.evoked[(unit.tag, "being_repaired_unit")] = repair_target
                    elif not repair_candidate.empty:
                        repair_target = repair_candidate.closest_to(unit)
                        actions.append(unit(AbilityId.EFFECT_REPAIR_MULE, repair_target))
                        self.evoked[(unit.tag, "being_repaired_unit")] = repair_target
                    else:
                        # 할게 없는 상태.
                        # 평소 대기 시에는 우리 커맨드보다 조금 안쪽에서 대기
                        if self.cc.position.x < 50:
                            actions.append(unit.move(Point2((self.cc.position.x - 5, self.cc.position.y))))
                        else:
                            actions.append(unit.move(Point2((self.cc.position.x + 5, self.cc.position.y))))
                        self.evoked[(unit.tag, "being_repaired_unit")] = None

        return actions

    def on_end(self, game_result):
        if self.sock is not None:
            score = 1. if game_result is Result.Victory else -1.
            self.sock.send_multipart((
                CommandType.SCORE,
                pickle.dumps(self.name),
                pickle.dumps(self.game_id),
                pickle.dumps(score)
            ))
            self.sock.recv_multipart()