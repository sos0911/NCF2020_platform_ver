
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'


# python -m bots.nc_example_v5.bot --server=172.20.41.105
# kill -9 $(ps ax | grep SC2_x64 | fgrep -v grep | awk '{ print $1 }')
# kill -9 $(ps ax | grep bots.nc_example_v5.bot | fgrep -v grep | awk '{ print $1 }')
# ps aux

import os
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pathlib
import pickle
import time

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint

from .consts import ArmyStrategy, CommandType, EconomyStrategy

INF = 1e9

nest_asyncio.apply()

# model architecture 정의
# forward() 정의
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5 + 12, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.vf = nn.Linear(64, 1)
        self.economy_head = nn.Linear(64, len(EconomyStrategy))
        self.army_head = nn.Linear(64, len(ArmyStrategy))

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        value = self.vf(x)
        economy_logp = torch.log_softmax(self.economy_head(x), -1)
        army_logp = torch.log_softmax(self.army_head(x), -1)
        # bz = ??
        bz = x.shape[0]
        logp = (economy_logp.view(bz, -1, 1) + army_logp.view(bz, 1, -1)).view(bz, -1)
        return value, logp


class Bot(sc2.BotAI):
    """
    example v1과 유사하지만, 빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    """
    def __init__(self, step_interval=5.0, host_name='', sock=None):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / 'model.pt'
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
            except Exception as exc:
                import traceback; traceback.print_exc()

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval
        self.evoked = dict()

        self.economy_strategy = EconomyStrategy.MARINE.value
        self.army_strategy = ArmyStrategy.DEFENSE

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색
        # (32.5, 31.5) or (95.5, 31.5)
        if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0:
            # self.enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치
            self.enemy_cc = Point2(Point2((95.5, 31.5)))  # 적 시작 위치
        else:
            self.enemy_cc = Point2(Point2((32.5, 31.5)))  # 적 시작 위치

        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        # data = (JOIN, game_id)
        # self.sock.send_multipart([pickle.dumps(d) for d in data])

    async def on_step(self, iteration: int):
        """

        """
        # 아래 action들이 이번 step에 할 수 있는 액션의 풀을 나타내는 건가..?
        actions = list() # 이번 step에 실행할 액션 목록

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

        actions += self.train_action()
        actions += self.unit_actions()
        await self.do_actions(actions)

    def set_strategy(self):
        #
        # 특징 추출
        #
        state = np.zeros(5 + len(EconomyStrategy), dtype=np.float32)
        state[0] = self.cc.health_percentage
        state[1] = min(1.0, self.minerals / 1000)
        state[2] = min(1.0, self.vespene / 1000)
        state[3] = min(1.0, self.time / 360)
        state[4] = min(1.0, self.state.score.total_damage_dealt_life / 2500)
        # 구조물이 아닌 유닛 for문??
        for unit in self.units.not_structure:
            state[5 + EconomyStrategy.to_index[unit.type_id]] += 1
        state = state.reshape(1, -1)

        # NN
        data = [
            CommandType.STATE,
            pickle.dumps(self.game_id),
            pickle.dumps(state.shape),
            state,
        ]
        # self.sock이 무슨 의미일까?
        # actor-critic에서 multuple actor?
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

        # 다음에 뭘 뽑을지 선택하는 것 = economy_strategy
        # 여기 왕중요!!!!!!!! 결국 RL이 학습한 action은 economy_strategy와 army_strategy을 결정.
        # 즉 다음에 뭘뽑을지랑 공격 타이밍만 RL로 학습하고 유닛별 컨트롤이나 그런건 Rule-based로 구성됨.
        economy_strategy = EconomyStrategy.to_type_id[action // len(ArmyStrategy)]
        army_strategy = ArmyStrategy(action % len(ArmyStrategy))
        return economy_strategy, army_strategy

    def train_action(self):
        #
        # 사령부 명령 생성
        #
        actions = list()
        next_unit = self.economy_strategy
        if self.can_afford(next_unit):
            if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                # cc : command center
                actions.append(self.cc.train(next_unit))
                self.evoked[(self.cc.tag, 'train')] = self.time
        return actions

    def unit_actions(self):
        #
        # 유닛 명령 생성
        #
        actions = list()
        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택
            enemy_unit = self.enemy_start_locations[0]
            # 알려진 enemy unit이 존재한다면?
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(self.enemy_cc) < unit.distance_to(enemy_unit):
                target = self.enemy_cc
            else:
                target = enemy_unit

            if unit.type_id is not UnitTypeId.MEDIVAC:
                if self.army_strategy is ArmyStrategy.OFFENSE:
                    # 전투가능한 유닛 수가 15를 넘으면 적 본진으로 공격
                    actions.append(unit.attack(target))
                else:  # ArmyStrategy.DEFENSE
                    # 적 사령부 방향에 유닛 집결
                    target = self.start_location + 0.25 * (self.enemy_cc.position - self.start_location)
                    actions.append(unit.attack(target))

                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if self.army_strategy is ArmyStrategy.OFFENSE and unit.distance_to(target) < 15:
                        # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time

            # 자기가 수송선일 때..
            if unit.type_id is UnitTypeId.MEDIVAC:
                if self.wounded_units.exists:
                    wounded_unit = self.wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    if self.combat_units.exists:
                        actions.append(unit.move(self.combat_units.center))
            
            # 시즈탱크라면?
            if unit.type_id is UnitTypeId.SIEGETANK 
                # 적이 보이고 사정거리 내에 있는데 붙어 있지 않으면 바로 시즈모드
                if self.known_enemy_units.exists :
                    # 시즈탱크가 공격 가능한 지상 유닛이어야 함
                    enemy_ground_units = self.known_enemy_units.not_flying
                    mindist = 500 # mindist
                    closest_pos = Point2()
                    for eunit in enemy_ground_units:
                        # 각 적 유닛간의 거리 조사
                        if mindist > unit.distance_to(eunit):
                            mindist = unit.distance_to(eunit)
                            closest_pos = eunit.position
                    
                    if mindist >= 6.0 and mindist <= 13.0:
                        actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                    
                    # 적이 날 공격하려 한다면 반대방향으로 튀자..
                    # 무빙샷 구현은 어떻게?
                    if mindist < 6.0:
                        dest = unit.position - eunit.position + unit.position
                        actions.append(unit.move(dest)))

            # 시즈모드 시탱
            if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                 if self.known_enemy_units.exists :
                    enemyunits = self.known_enemy_units
                    mindist = 500 # mindist
                    closest_pos = Point2()
                    for eunit in enemyunits:
                        if eunit.can_attack_ground == False:
                            continue
                        # 각 적 유닛간의 거리 조사
                        if mindist > unit.distance_to(eunit):
                            mindist = unit.distance_to(eunit)
                            closest_pos = eunit.position
                    # 너무 붙었으면 바로 시즈모드 해제
                    if mindist < 6.0:
                        actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))


            # 사신은 쓰지 않을 것이므로 제외

            # 토르라면?
            # default는 폭약 모드(경장갑)
            # morph로 고폭탄 모드로 변경 가능
            # THOR(폭약), THORAP(고폭탄)
            if unit.type_id is UnitTypeId.THOR:
                # 전투순양함이 사거리 내에 들어오면 바로 고폭탄 모드로 변경
                enemy_inrange_units = self.known_enemy_units.in_attack_range_of(unit)
                for eunit in enemy_inrange_units:
                    if eunit.type_id is BATTLECRUISER:
                        actions.append(unit(MORPH_THORHIGHIMPACTMODE))
                        break

            if unit.type_id is UnitTypeId.THORAP:
                # 전투순양함이 시야나 사거리 내에서 사라지면 다시 디폴트로 변경
                enemy_inrange_units = self.known_enemy_units.in_attack_range_of(unit)
                batcheck = False
                # 사정거리 내 배틀크루저 중 가장 hp가 적은 놈을 집중사격
                closest_battle = None
                minHP_battle = INF
                for eunit in enemy_inrange_units:
                    if eunit.type_id is BATTLECRUISER:
                        if batcheck is False:
                            batcheck = True
                        if minHP_battle > eunit.health
                            minHP_battle = eunit.health
                            closest_battle = eunit

                if batcheck is True:
                    actions.append(unit.attack(closest_battle))
                else:
                    actions.append(unit(MORPH_THOREXPLOSIVEMODE))

            # 토르가 공성전차에겐 약하다.. 이속이 느려서
            # 이것에 대한 코드도 추가예정


            # 바이킹 전투기 모드(공중)
            if unit.type_id is UnitTypeId.VIKINGASSAULT:
                # 무빙샷이 필요
                # 한방이 쎈 유닛이다.
                # 사정거리 내면 치고 빠지기
                temp = self.known_enemy_units.in_attack_range_of(unit)
                enemy_flying_unit = temp.flying
                if enemy_flying_unit.amount > 0:
                    target = None
                    targetHP = INF
                    targetPos = Point2()
                    for eunit in enemy_flying_unit:
                        if eunit.health < targetHP:
                            target = eunit
                            targetHP = eunit.health
                            targetPos = eunit.position
                    
                    actions.append(unit.attack(target))
                    # 도망가기
                    # dir normalize도 있나..?
                    movdir = unit.position - target.position
                    actions.append(unit.move(unit.position + movdir*3))
                
                # 커맨드 주위에 아무도 없을 때 변신해서 커맨드 부순다.
                temp2 = self.known_enemy_structures.in_attack_range_of(unit)
                cc_of_enemy = self.units(COMMANDCENTER)
                if temp.amount == 0 and cc_of_enemy.first in temp2:
                    actions.append(unit(MORPH_VIKINGFIGHTERMODE))
                    actions.append(unit.attack(self.enemy_cc))


            # 바이킹 전투 모드(지상)
            if unit.type_id is UnitTypeId.VIKINGFIGHTER:
                # 탱크가 일정 거리 내에 있으면(시즈 모드 거리에) 전투기 변환
                enemy_tanks = self.known_enemy_units.filter(lambda u : u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                for eunit in enemy_tanks:
                    if eunit.ground_range >= unit.distance_to(eunit):
                        actions.append(unit(MORPH_VIKINGASSAULTMODE))
                        break
               


            # 밴시? 
            # 공성전차를 위주로 잡게 한다.
            # 기본적으로 공성전차를 찾아다니되 들키면 튄다 ㅎㅎ
            if unit.type_id is UnitTypeId.BANSHEE:
                # 마나를 조금 채우고 움직인다.
                if unit.energy_percentage >= 0.5:
                    enemy_tanks = self.known_enemy_units.filter(lambda u : u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                    target = enemy_tanks.closest_to(unit.position)
                    actions.append(unit.attack(target))
                
                # 만약 주위에 자신을 공격 가능한 유닛이 있다면 클로킹에 들어간다.
                # 마나가 없거나 들키면 도망간다
                threats = self.known_enemy_units.filter(lambda u : u.can_attack_air and u.air_range >= unit.distance_to(u))
                if threats.amount > 0:
                    if unit.cloak == Unit.CloakState.NotCloaked:
                        actions.append(unit(BEHAVIOR_CLOAKON_BANSHEE))
                    else if unit.cloak == Unit.CloakState.CloakedDetected or unit.energy_percentage < 0.2:
                        closest_threat = threats.closest_to(unit.position)
                        actions.append(unit.move((unit.position - closest_threat.position)*3 + unit.position))

                # 만약 주위에 아무도 자길 때릴 수 없으면 클락을 풀어 마나보충
                else if threats.amount == 0 and unit.cloak == Unit.CloakState.Cloaked:
                    actions.append(unit(BEHAVIOR_CLOAKOFF_BANSHEE))
            
            # 이제 남은거..
            # 부대지정을 어떻게 할까..?


        return actions

    def on_end(self, game_result):
        if self.sock is not None:
            # 신기한 문법..p
            # score = 각 trajectory에 대한 reward 용도...?
            score = 1. if game_result is Result.Victory else -1.
            self.sock.send_multipart((
                CommandType.SCORE, 
                pickle.dumps(self.game_id),
                pickle.dumps(score),
            ))
            self.sock.recv_multipart()
