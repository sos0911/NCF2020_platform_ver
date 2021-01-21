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

# .을 const 앞에 왜 찍는 거지?
from .consts import ArmyStrategy, CommandType, EconomyStrategy

INF = 1e9

nest_asyncio.apply()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # wonseok add #
        self.fc1 = nn.Linear(5 + len(EconomyStrategy) * 2 + 1, 128)
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

    def __init__(self, step_interval=5.0, host_name='', sock=None, version=""):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        ## donghyun edited ##
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / ('model' + version + '.pt')
                #self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'))) # gpu
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # cpu
            except Exception as exc:
                import traceback;
                traceback.print_exc()
        ## donghyun end ##


    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval
        self.evoked = dict()
        self.enemy_exists = dict()

        # 현재 병영생산전략
        self.economy_strategy = EconomyStrategy.MARINE.value
        # 현재 군대전략
        self.army_strategy = ArmyStrategy.DEFENSE

        # offense mode?
        self.offense_mode = False
        # 핵 보유?
        self.has_nuke = False
        self.map_height = 63
        self.map_width = 128
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색
        # 내 그룹 initiate
        self.my_groups = []

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
        self.left_down = Point2((12.5, 10.5))
        self.right_down = Point2((110.5, 10.5))
        self.left_up = Point2((12.5, 51.5))
        self.right_up = Point2((110.5, 51.5))

        # 밤까마귀 cache
        self.first_enemy_Raven = None
        # 아군 로봇에게 수리받는 중인 애들 list cache
        self.being_repaired_units_list = []

        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        # data = (JOIN, game_id)
        # self.sock.send_multipart([pickle.dumps(d) for d in data])

    async def on_unit_destroyed(self, unit_tag):
        """ Override this in your bot class. """
        self.enemy_exists.pop(unit_tag, None)

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

        self.cached_known_enemy_units = self.known_enemy_units()
        self.cached_known_enemy_structures = self.known_enemy_structures()
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first # 왠지는 모르겠는데 이걸 추가해야 실시간 tracking이 된다..

        # 공격 모드가 아닌 기타 모드일때
        # offense_mode가 될지 말지 정함
        # 하나라도 트리거가 된다면 모두 트리거가 된다.
        for unit in self.units.not_structure:
            if self.select_mode(unit):
                # for state
                self.offense_mode = True
                # 모든 유닛이 트리거 작동
                # 정찰 중인 벌쳐만 빼고..
                for unit in self.units.not_structure:
                    if self.evoked.get(("scout_unit_tag"), None) is not None and unit.tag == self.evoked.get(
                            ("scout_unit_tag")):
                        self.evoked[(unit.tag, "offense_mode")] = False
                        continue
                    self.evoked[(unit.tag, "offense_mode")] = True
                break
            else:
                self.offense_mode = False

        # 공중 공격 가능 유닛 우선순위 1순위 target!
        # 만약 아군 밴시가 detected되었다면
        # 가장 가까운 밤까마귀 cache
        self.first_enemy_Raven = None
        for our_unit in self.units:
            if our_unit.type_id is UnitTypeId.BANSHEE and our_unit.is_revealed:
                enemy_ravens = self.known_enemy_units.filter(lambda u: u.type_id is UnitTypeId.RAVEN)
                if not enemy_ravens.empty:
                    self.first_enemy_Raven = enemy_ravens.sorted(lambda u: our_unit.distance_to(u))[0]
                    break

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

        # 적 reaper state를 기록하기 위한 +1
        state = np.zeros(5 + (len(EconomyStrategy) * 2) + 1, dtype=np.float32)
        state[0] = self.cc.health_percentage
        state[1] = min(1.0, self.minerals / 1000)
        state[2] = min(1.0, self.vespene / 1000)
        state[3] = min(1.0, self.time / 360)
        state[4] = min(1.0, self.state.score.total_damage_dealt_life / 2500)
        for unit in self.units.not_structure:
            if unit.type_id is UnitTypeId.THORAP:
                state[5 + EconomyStrategy.to_index[EconomyStrategy.THOR.value]] += 1
            elif unit.type_id is UnitTypeId.VIKINGASSAULT:
                state[5 + EconomyStrategy.to_index[EconomyStrategy.VIKINGFIGHTER.value]] += 1
            elif unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                state[5 + EconomyStrategy.to_index[EconomyStrategy.SIEGETANK.value]] += 1
            else:
                state[5 + EconomyStrategy.to_index[unit.type_id]] += 1

        state[5 + len(EconomyStrategy) - 1] = self.has_nuke

        # wonseok add #
        for type_id in self.enemy_exists.values():
            if type_id is UnitTypeId.THORAP:
                state[5 + len(EconomyStrategy) + EconomyStrategy.to_index[EconomyStrategy.THOR.value]] += 1
            elif type_id is UnitTypeId.VIKINGASSAULT:
                state[5 + len(EconomyStrategy) + EconomyStrategy.to_index[EconomyStrategy.VIKINGFIGHTER.value]] += 1
            elif type_id is UnitTypeId.SIEGETANKSIEGED:
                state[5 + len(EconomyStrategy) + EconomyStrategy.to_index[EconomyStrategy.SIEGETANK.value]] += 1
            else:
                state[5 + len(EconomyStrategy) + EconomyStrategy.to_index[type_id]] += 1

        # 적이 핵을 갖고 있는지 안 갖고 있는지는 알 방법이 없다.
        # 0으로 세팅.
        # state[5 + len(EconomyStrategy) * 2 - 1] = False

        # offense_mode
        state[5 + len(EconomyStrategy) * 2] = self.offense_mode

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
        next_unit = self.economy_strategy
        # 핵
        if next_unit == EconomyStrategy.NUKE.value:
            if self.can_afford(AbilityId.BUILD_NUKE) and not self.has_nuke and self.time - self.evoked.get(
                    (self.cc.tag, 'train'), 0) > 1.0:
                actions.append(self.cc(AbilityId.BUILD_NUKE))
                self.has_nuke = True
                self.evoked[(self.cc.tag, 'train')] = self.time
        # 지게로봇
        elif next_unit == EconomyStrategy.MULE.value:
            if await self.can_cast(self.cc, AbilityId.CALLDOWNMULE_CALLDOWNMULE, only_check_energy_and_cooldown=True):
                if self.cc.position.x < 50:
                    mule_summon_point = Point2((self.cc.position.x - 5, self.cc.position.y))
                else:
                    mule_summon_point = Point2((self.cc.position.x + 5, self.cc.position.y))
                    # 정해진 곳에 MULE 소환
                actions.append(self.cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mule_summon_point))
        # 나머지
        elif self.can_afford(next_unit):
            if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(self.cc.train(next_unit))
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
                lambda u: u.can_attack_air and u.air_range + 2 >= unit.distance_to(u))
            for eunit in self.known_enemy_units:
                if eunit.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(eunit):
                    threats.append(eunit)
        else:
            threats = self.known_enemy_units.filter(
                lambda u: u.can_attack_ground and u.ground_range + 2 >= unit.distance_to(u))
            for eunit in self.known_enemy_units:
                if eunit.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(eunit):
                    threats.append(eunit)

        return threats

    def select_mode(self, unit: Unit):
        # 정찰중인 벌쳐는 제외
        if unit.tag == self.evoked.get(("scout_unit_tag")):
            self.evoked[(unit.tag, "offense_mode")] = False
            return False
        # 방어모드일때 공격모드로 전환될지 트리거 세팅
        # 방어모드라면 False, 공격모드로 바뀐다면 True return
        nearby_enemies = self.known_enemy_units.filter(
            lambda u: unit.distance_to(u) <= max(unit.sight_range, unit.ground_range, unit.air_range))
        if nearby_enemies.empty:
            self.evoked[(unit.tag, "offense_mode")] = False
            return False
        else:
            self.evoked[(unit.tag, "offense_mode")] = True
            return True

    # 무빙샷
    def moving_shot(self, actions, unit, cooldown, target_func, margin_health: float = 0, minimum: float = 0):
        # print("WEAPON COOLDOWN : ", unit.weapon_cooldown)
        if unit.weapon_cooldown < cooldown:
            target = target_func(unit)
            if self.time - self.evoked.get((unit.tag, "COOLDOWN"), 0.0) >= minimum:
                actions.append(unit.attack(target))
                self.evoked[(unit.tag, "COOLDOWN")] = self.time

        elif (margin_health == 0 or unit.health_percentage <= margin_health) and self.time - self.evoked.get(
                (unit.tag, "COOLDOWN"), 0.0) >= minimum:  # 무빙을 해야한다면
            maxrange = 0
            total_move_vector = Point2((0, 0))
            showing_only_enemy_units = self.known_enemy_units.not_structure.filter(lambda e: e.is_visible)

            # 자신이 클라킹 상태가 아닐 때나 클라킹 상태이지만 발각됬을 때
            if not unit.is_cloaked or unit.is_revealed:
                if not unit.is_flying:
                    # 배틀크루저 예외처리.
                    # 배틀은 can_attack_air/ground와 무기 범위가 다 false, 0이다.
                    threats = showing_only_enemy_units.filter(lambda u: ((u.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(u)) or (
                            u.can_attack_ground and u.ground_range + 2 >= unit.distance_to(u))))
                    for eunit in threats:
                        if eunit.type_id is UnitTypeId.BATTLECRUISER:
                            maxrange = max(maxrange, 6)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (6 + 2 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                        else:
                            maxrange = max(maxrange, eunit.ground_range)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (eunit.ground_range + 2 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                else:
                    threats = showing_only_enemy_units.filter(
                        lambda u: ((u.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(u)) or (
                                u.can_attack_air and u.air_range + 2 >= unit.distance_to(u))))
                    for eunit in threats:
                        if eunit.type_id is UnitTypeId.BATTLECRUISER:
                            maxrange = max(maxrange, 6)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (6 + 2 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector
                        else:
                            maxrange = max(maxrange, eunit.air_range)
                            move_vector = unit.position - eunit.position
                            move_vector /= (math.sqrt(move_vector.x ** 2 + move_vector.y ** 2))
                            move_vector *= (eunit.air_range + 2 - unit.distance_to(eunit)) * 1.5
                            total_move_vector += move_vector

                if not threats.empty:
                    total_move_vector /= math.sqrt(total_move_vector.x ** 2 + total_move_vector.y ** 2)
                    total_move_vector *= maxrange
                    # 이동!
                    dest = Point2((self.clamp(unit.position.x + total_move_vector.x, 0, self.map_width),
                                   self.clamp(unit.position.y + total_move_vector.y, 0, self.map_height)))
                    actions.append(unit.move(dest))

        return actions

    # 유닛 그룹 정하기
    # 시즈탱크 제외하고 산정.
    # 그룹당 중복 가능..
    def unit_groups(self):
        groups = []
        center_candidates = self.units.not_structure.filter(
            lambda u: u.type_id is not UnitTypeId.SIEGETANKSIEGED and u.type_id is not UnitTypeId.SIEGETANK)
        for unit in center_candidates:
            group = center_candidates.closer_than(5, unit)
            groups.append(group)

        groups.sort(key=lambda g: g.amount, reverse=True)
        ret_groups = []

        # groups가 비는 경우는 시즈탱크 제외 유닛이 아예 없다는 것
        # 이 경우 빈 list 반환
        if not groups:
            return ret_groups

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

            if unit.type_id is not (UnitTypeId.MEDIVAC and UnitTypeId.RAVEN and UnitTypeId.SIEGETANK and UnitTypeId.SIEGETANKSIEGED and \
                    UnitTypeId.MULE) and not self.evoked.get((unit.tag, "offense_mode"),False):
                if unit.type_id is UnitTypeId.HELLION and unit.tag == self.evoked.get(("scout_unit_tag")):
                    pass
                elif self.army_strategy is ArmyStrategy.DEFENSE:
                    move_position = self.cc.position
                    # move와 attack 둘 중 뭐가 나을까..?
                    actions.append(unit.attack(move_position))
                elif self.army_strategy is ArmyStrategy.READY_LEFT:
                    actions.append(unit.attack(self.ready_left))
                
                elif self.army_strategy is ArmyStrategy.READY_CENTER:
                    actions.append(unit.attack(self.ready_center))

                elif self.army_strategy is ArmyStrategy.READY_RIGHT:
                    actions.append(unit.attack(self.ready_right))

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
                            yamato_candidate_id = [UnitTypeId.THORAP, UnitTypeId.THOR, UnitTypeId.BATTLECRUISER,
                                                   UnitTypeId.SIEGETANKSIEGED,
                                                   UnitTypeId.SIEGETANK, UnitTypeId.RAVEN] if self.first_enemy_Raven is None else \
                                [UnitTypeId.RAVEN, UnitTypeId.THORAP, UnitTypeId.THOR, UnitTypeId.BATTLECRUISER,
                                 UnitTypeId.SIEGETANKSIEGED,
                                 UnitTypeId.SIEGETANK]

                            for eunit_id in yamato_candidate_id:
                                target_candidate = self.known_enemy_units.filter(
                                    lambda u: u.type_id is eunit_id and unit.distance_to(u) <= yamato_enemy_range)
                                target_candidate.sorted(lambda u: u.health, reverse=True)
                                if not target_candidate.empty:
                                    return target_candidate.first

                            # 위 리스트 안 개체들이 없다면 나머지 중 타겟팅
                            # 나머지 유닛도 없다면 적 커맨드로 ㄱㄱ
                            enemy_left = self.known_enemy_units.filter(
                                lambda u: unit.distance_to(u) <= yamato_enemy_range)
                            enemy_left.sorted(lambda u: u.health, reverse=True)
                            if not enemy_left.empty:
                                return enemy_left.first
                            else:
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
                                for eunit in self.cached_known_enemy_units:
                                    if eunit.is_visible and eunit.distance_to(unit) < min_dist:
                                        target = eunit
                                        min_dist = eunit.distance_to(unit)
                            return target

                        enemy_inrange_units = self.cached_known_enemy_units.in_attack_range_of(unit)
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
                            enemy_inrange_units = self.cached_known_enemy_units.in_attack_range_of(unit)
                            enemy_flying_heavy = enemy_inrange_units.filter(lambda u: u.is_armored)

                            if self.known_enemy_units.empty:
                                target = self.enemy_cc
                            elif enemy_flying_heavy.empty:
                                target = self.known_enemy_units.closest_to(unit)
                            else:
                                target = enemy_flying_heavy.sorted(lambda e: e.health)[0]
                            return target

                        enemy_inrange_units = self.cached_known_enemy_units.in_attack_range_of(unit)
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
                    # desired_vector 관련된 정보는 모드에 관계없이 항상 갱신.
                    # default : cc.position
                    desired_pos = self.cc.position

                    if self.my_groups:
                        # 만약 첫 프레임이거나 이전 프레임에 설정된 그룹 센터와 현재 계산된 그룹 센터가 일정 거리 이상(5)다르다면 이동
                        if self.evoked.get((unit.tag, "desire_add_vector"), None) is None:
                            dist = random.randint(5, 9)
                            dist_x = random.randint(2, dist)
                            dist_y = math.sqrt(dist ** 2 - dist_x ** 2) if random.randint(0, 1) == 0 else -math.sqrt(
                                dist ** 2 - dist_x ** 2)
                            desire_add_vector = Point2((-dist_x, dist_y)) if self.cc.position.x < 50 else Point2((dist_x, dist_y))
                            desired_pos = self.my_groups[0].center + desire_add_vector
                            desired_pos = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                                  self.clamp(desired_pos.y, 0, self.map_height)))
                            self.evoked[(unit.tag, "group_center")] = self.my_groups[0].center
                            self.evoked[(unit.tag, "desire_add_vector")] = desire_add_vector
                        else:
                            if self.my_groups[0].center.distance_to(
                                    self.evoked.get((unit.tag, "group_center"), self.cc.position)) > 7:
                                self.evoked[(unit.tag, "group_center")] = self.my_groups[0].center
                            desired_pos = self.evoked.get((unit.tag, "group_center"), self.cc.position) + self.evoked.get(
                                (unit.tag, "desire_add_vector"), None)

                    # 시즈탱크는 공격, 방어 상관없이 기본적으로 항상 정해진 그룹 센터 주변으로 포지셔닝
                    # 그룹 센터에서 상대적으로 뒤쪽에 대기한다.
                    # 그룹 센터에서 거리는 왼쪽으로 랜덤으로 정해지되, 5-9 정도.
                    # 근처 위협이 있다면 무빙샷
                    def target_func(unit):
                        selected_enemies = []
                        if self.cached_known_enemy_units.not_structure.exists:
                            selected_enemies = self.cached_known_enemy_units.not_structure.filter(
                                lambda u: u.is_visible and not u.is_flying)
                        if selected_enemies.empty:
                            return self.enemy_cc
                        else:
                            return selected_enemies.closest_to(unit)

                    threats = self.select_threat(unit)

                    # 주위에 자신을 노리는 애들이 없는 경우는 항상 desired_pos로 이동
                    # desired_pos 근처에 도달하면 시즈모드.
                    # 자신을 노리는 애들이 있다면 해당 상대를 향햐여 무빙샷
                    if threats.empty:
                        if int(unit.position.x) == int(desired_pos.x) and int(unit.position.y) == int(
                                desired_pos.y):
                            # 근처에 위협이 없고 도착 지점 근처에 다다랐으면 시즈모드.
                            actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                        else:
                            actions.append(unit.attack(desired_pos))
                    # 만약 있다면 시즈모드 말고 무빙샷
                    else:
                        actions = self.moving_shot(actions, unit, 3, target_func)

                # 시즈모드 시탱
                # 시즈모드일 때에도 공격, 방어모드 상관없이 작동한다.
                if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                    if self.known_enemy_units.not_structure.exists:
                        # 타겟팅 정하기
                        target = None
                        # HP 적은 애를 타격하지만, 중장갑 위주
                        targets = self.known_enemy_units.not_structure.filter(
                            lambda u: not u.is_flying and unit.distance_to(u) <= unit.ground_range)
                        armored_targets = targets.filter(lambda u: u.is_armored)
                        light_targets = targets - armored_targets
                        if not armored_targets.empty:
                            target = armored_targets.sorted(lambda u: u.health)[0]
                        elif not light_targets.empty:
                            target = light_targets.sorted(lambda u: u.health)[0]

                        if target is not None:
                            actions.append(unit.attack(target))

                        # 시즈모드 풀지 안풀지 결정하기
                        # 나와있는 ground_range에 0.7쯤 더해야 실제 사정거리가 된다..
                        # 넉넉잡아 2로..
                        threats = self.select_threat(unit)
                        # # 한 유닛이라도 자신을 때릴 수 있으면 바로 시즈모드 해제
                        # if threats.amount > 0:
                        #     actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                        for eunit in threats:
                            if unit.distance_to(eunit) < 3.5:
                                actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                                break

                    # 어느 때라도 상관없이 그룹 센터가 일정 거리 이상(10)달라지면 시즈모드 풀기(이동 준비)
                    if self.my_groups:
                        if self.my_groups[0].center.distance_to(self.evoked.get((unit.tag, "group_center"))) > 7:
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
                            targets = self.cached_known_enemy_units.not_structure.not_flying
                            if targets.empty:
                                return self.enemy_cc # point2

                            light_targets = targets.filter(lambda u: u.is_light)
                            # 경장갑이 타겟 중에 없으니 나머지 중 가장 가까운 놈 때리기
                            if light_targets.empty:
                                return targets.sorted(lambda u: unit.distance_to(u))[0]
                            # 경장갑
                            else:
                                return light_targets.sorted(lambda u: unit.distance_to(u))[0]

                        actions = self.moving_shot(actions, unit, 10, target_func)
                    # 내가 정찰용 화염차라면?
                    elif unit.tag == self.evoked.get(("scout_unit_tag")) and self.time - self.evoked.get((unit.tag, "end_time"), 0.0) >= 5.0:

                            if self.evoked.get((unit.tag, "scout_routine"), []) == [] :
                                self.evoked[(unit.tag, "scout_routine")] = ["Center", "RightUp", "RightDown", "LeftDown", "LeftUp", "End"]

                            for eunit in self.cached_known_enemy_units:
                                if unit.distance_to(eunit) <= unit.sight_range-2: # sight : 10, attack : 5
                                    if self.army_strategy == ArmyStrategy.DEFENSE :
                                        actions.append(unit.move(self.cc))
                                    elif self.army_strategy == ArmyStrategy.READY_LEFT :
                                        actions.append(unit.move(self.ready_left))
                                    elif self.army_strategy == ArmyStrategy.READY_CENTER :
                                        actions.append(unit.move(self.ready_center))
                                    elif self.army_strategy == ArmyStrategy.READY_RIGHT :
                                        actions.append(unit.move(self.ready_right))
                                    break

                            # 적, 아군 통틀어 어떤 유닛, 건물이라도 만나면 정찰 방향 수정
                            other_units = self.units - {unit}
                            if (unit.is_idle or other_units.closer_than(4, unit).exists or self.known_enemy_units.closer_than(unit.sight_range, unit).exists) \
                                    and self.time - self.evoked.get((unit.tag, "scout_time"), 0) >= 3.0 :
                                next = self.evoked.get((unit.tag, "scout_routine"))[0]

                                if next == "Center":
                                    actions.append(unit.move(self.enemy_cc))
                                    self.evoked[(unit.tag, "scout_routine")] = self.evoked.get((unit.tag, "scout_routine"))[1:]

                                elif next == "RightUp" :
                                    if abs(unit.position.y - 31.5) > 5.0 :
                                        actions.append(unit.move(self.right_up))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get((unit.tag, "scout_routine"))[1:]
                                    else :
                                        actions.append(unit.move(Point2((unit.position.x, self.right_up.y))))
                                        

                                elif next == "RightDown" :
                                    if abs(unit.position.y - 31.5) > 5.0 :
                                        actions.append(unit.move(self.right_down))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get((unit.tag, "scout_routine"))[1:]
                                    else :
                                        actions.append(unit.move(Point2((unit.position.x, self.right_down.y))))

                                elif next == "LeftDown" :
                                    if abs(unit.position.y - 31.5) > 5.0 :
                                        actions.append(unit.move(self.left_down))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get((unit.tag, "scout_routine"))[1:]
                                    else :
                                        actions.append(unit.move(Point2((unit.position.x, self.left_down.y))))

                                elif next == "LeftUp" :
                                    if abs(unit.position.y - 31.5) > 5.0 :
                                        actions.append(unit.move(self.left_up))
                                        self.evoked[(unit.tag, "scout_routine")] = self.evoked.get((unit.tag, "scout_routine"))[1:]
                                    else :
                                        actions.append(unit.move(Point2((unit.position.x, self.left_up.y))))

                                elif next == "End" :
                                    if self.army_strategy == ArmyStrategy.DEFENSE :
                                        actions.append(unit.move(self.cc))
                                    elif self.army_strategy == ArmyStrategy.READY_LEFT :
                                        actions.append(unit.move(self.ready_left))
                                    elif self.army_strategy == ArmyStrategy.READY_CENTER :
                                        actions.append(unit.move(self.ready_center))
                                    elif self.army_strategy == ArmyStrategy.READY_RIGHT :
                                        actions.append(unit.move(self.ready_right))

                                    self.evoked[(unit.tag), "end_time"] = self.time
                                    self.evoked[(unit.tag, "scout_routine")] = []

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
                        # 우선순위 1
                        # 눈에 보이는(visible) 공중 유닛 대상
                        first_targets = self.known_enemy_units.filter(lambda e: e.is_flying and e.is_visible)
                        if not first_targets.empty:

                            self.evoked["Last_enemy_aircraft_time"] = self.time

                            def target_func(unit):

                                target = first_targets.sorted(lambda e: e.health)[0]
                                return target

                            actions = self.moving_shot(actions, unit, 15, target_func)

                        # 우선순위 2
                        else:
                            # # 우선순위 2로 이행
                            # sidged_targets = self.known_enemy_units.filter(
                            #     lambda u: u.type_id is UnitTypeId.SIEGETANKSIEGED)
                            # tank_targets = self.known_enemy_units.filter(lambda u: u.type_id is UnitTypeId.SIEGETANK)
                            #
                            # if sidged_targets.amount > 0:
                            #     target = sidged_targets.sorted(lambda e: e.health)[0]
                            #     actions.append(unit.move(target))
                            #
                            #     if unit.distance_to(target) <= 3.0:
                            #         actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))
                            #
                            # elif tank_targets.amount > 0:
                            #     target = tank_targets.sorted(lambda e: e.health)[0]
                            #     actions.append(unit.move(target))
                            #
                            #     if unit.distance_to(target) <= 3.0:
                            #         actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))

                            # 알려진 지상 유닛과 커맨드 중 가장 HP가 없는 걸 때리기
                            # 적 공중 유닛이 2초 이상 보이지 않는 경우에만 해당
                            # 유닛 그룹 중앙에서 내려서 싸울 것.
                            targets = self.known_enemy_units.filter(
                                lambda u: unit.sight_range > unit.distance_to(u))
                            # 일정 시간 이상 적 공중 유닛이 보이지 않고 그룹 센터로부터 일정 범위 안에 들어온다면 착륙(3)
                            if not targets.empty and self.time - self.evoked.get("Last_enemy_aircraft_time", 0.0) >= 2.0\
                                    and unit.distance_to(self.my_groups[0].center) < 3.0:
                                landing_loc = self.evoked.get((unit.tag, "landing_loc"), None)
                                if landing_loc is None or not await \
                                        self.can_place(building=AbilityId.MORPH_VIKINGASSAULTMODE, position=landing_loc):
                                    # loc = await self.find_placement(building=AbilityId.MORPH_VIKINGASSAULTMODE, near=self.my_groups[0].center,
                                    #                             max_distance=20)
                                    dist = random.randint(4, 6)
                                    dist_x = random.randint(3, dist)
                                    dist_y = math.sqrt(dist ** 2 - dist_x ** 2)\
                                        if random.randint(0,1) == 0 else -math.sqrt(dist ** 2 - dist_x ** 2)
                                    desire_add_vector = Point2((-dist_x, dist_y))
                                    desired_pos = self.my_groups[0].center + desire_add_vector
                                    landing_loc = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                                          self.clamp(desired_pos.y, 0, self.map_height)))
                                    self.evoked[(unit.tag, "landing_loc")] = landing_loc
                                actions.append(unit.move(landing_loc))
                                if unit.distance_to(landing_loc) < 2.0:
                                    actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))
                            else:
                                # 타깃이 없을 때 그룹 센터에서 대기하기 위한 코드
                                actions.append(unit.move(self.my_groups[0].center))


                # 바이킹 전투 모드(지상)
                # 공격 우선순위 : 공중유닛 > 사정거리 내 탱크 > 지상유닛 > 적 커맨드
                if unit.type_id is UnitTypeId.VIKINGASSAULT:

                    enemy_air = self.known_enemy_units.flying
                    ground_enemy_units = self.cached_known_enemy_units.not_structure.not_flying

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
                            ground_units = self.cached_known_enemy_units.not_flying.filter(
                                lambda e: e.is_visible)

                            if not ground_units.empty:
                                enemy_machines = self.known_enemy_units.filter(lambda u: u.is_visible and u.is_mechanical)
                                if enemy_machines.empty:
                                    return ground_units.closest_to(unit)
                                else:
                                    return enemy_machines.closest_to(unit)

                            return self.enemy_cc

                        if not ground_enemy_units.empty:
                            actions = self.moving_shot(actions, unit, 1, target_func, 0.5)

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
                            ground_units = self.cached_known_enemy_units.not_flying.filter(
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
                if unit.distance_to(
                        target) < 15 and unit.energy > 75 and self.army_strategy is ArmyStrategy.OFFENSE:  # 적들이 근처에 있고 마나도 있으면
                    known_only_enemy_units = self.known_enemy_units.not_structure
                    if known_only_enemy_units.exists:  # 보이는 적이 있다면
                        enemy_amount = known_only_enemy_units.amount
                        not_antiarmor_enemy = known_only_enemy_units.filter(
                            # anitarmor가 아니면서 가능대상에서 1초가 지났을 때...
                            lambda unit: self.time - self.evoked.get((unit.tag, "ANTIARMOR_POSSIBLE"), 0) >= 1.0
                            # (not unit.has_buff(BuffId.RAVENSHREDDERMISSILEARMORREDUCTION)) and
                        )
                        not_antiarmor_enemy_amount = not_antiarmor_enemy.amount

                        if not_antiarmor_enemy_amount / enemy_amount > 0.5:  # 안티아머 걸리지 않은게 절반 이상이면 미사일 쏘기 center에
                            enemy_center = not_antiarmor_enemy.center
                            select_unit = not_antiarmor_enemy.closest_to(enemy_center)
                            possible_units = known_only_enemy_units.closer_than(3, select_unit)  # 안티아머 걸릴 수 있는 애들

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
                else:  # 적들이 없으면
                    if self.units.not_structure.exists:  # 전투그룹 중앙 대기
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
                        # 1.시즈탱크
                        query_units = self.cached_known_enemy_units.filter(lambda
                                                                               u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED).sorted(
                            lambda u: u.health + unit.distance_to(u))
                        if not query_units.empty:
                            return query_units.first
                        # 2.토르
                        query_units = self.cached_known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.THOR or u.type_id is UnitTypeId.THORAP).sorted(
                            lambda u: u.health + unit.distance_to(u))
                        if not query_units.empty:
                            return query_units.first
                        # 3.가까운거
                        query_units = self.known_enemy_units.filter(lambda e: e.is_visible)
                        if not query_units.empty:
                            return query_units.closest_to(unit)
                        # 4.커맨드
                        # 이때는 None 리턴
                        return None

                    if target_func(unit) is None:
                        actions.append(unit.attack(self.enemy_cc))
                    else:
                        actions = self.moving_shot(actions, unit, 3, target_func, 0.5)


            # 밴시
            # 공성전차를 위주로 잡게 한다.
            # 기본적으로 공성전차를 찾아다니되 들키면 튄다 ㅎㅎ
            if unit.type_id is UnitTypeId.BANSHEE:

                # 아래 내용은 공격 정책이거나 offense mode가 아닐 시에도 항시 적용됨
                threats = self.select_threat(unit)
                # clock_threats = self.cached_known_enemy_units.filter(
                #     lambda u: u.type_id is UnitTypeId.RAVEN and unit.distance_to(u) <= u.sight_range)

                # 근처에 위협이 존재할 시 클라킹
                if not threats.empty and not unit.has_buff(BuffId.BANSHEECLOAK) and unit.energy_percentage >= 0.3:
                    actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_BANSHEE))

                # 만약 주위에 아무도 자길 때릴 수 없으면 클락을 풀어 마나보충
                if not threats.empty:
                    self.evoked[(unit.tag, "BANSHEE_CLOAK")] = self.time

                if threats.empty and self.time - self.evoked.get((unit.tag, "BANSHEE_CLOAK"), 0.0) >= 10 \
                        and unit.has_buff(BuffId.BANSHEECLOAK):
                    actions.append(unit(AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE))

                # 공격 정책이거나 offense mode가 트리거됬을 시
                if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):

                    def target_func(unit):
                        enemy_tanks = self.cached_known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                        if enemy_tanks.amount > 0:
                            target = enemy_tanks.closest_to(unit.position)
                            return target
                        # 만약 탱크가 없다면 HP가 가장 적으면서 가까운 아무 지상 유닛이나, 그것도 없다면 커맨드 직행
                        else:
                            targets = self.cached_known_enemy_units.filter(lambda u: u.type_id is not UnitTypeId.COMMANDCENTER and not u.is_flying)
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
                pickle.dumps(self.game_id),
                pickle.dumps(score),
            ))
            self.sock.recv_multipart()