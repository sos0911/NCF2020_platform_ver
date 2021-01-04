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


class Bot(sc2.BotAI):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def on_start(self):
        self.evoked = dict()
        self.army_strategy = ArmyStrategy.OFFENSE
        self.map_height = 63
        self.map_width = 128
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색
        # (32.5, 31.5) or (95.5, 31.5)
        if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0:
            # self.enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치
            self.enemy_cc = Point2(Point2((95.5, 31.5)))  # 적 시작 위치
        else:
            self.enemy_cc = Point2(Point2((32.5, 31.5)))  # 적 시작 위치

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인지를 인자로 넘겨 줌

        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        # 왜인지 모르겠지만, self.cc를 on_start에서 선언한 것으로는 인식이 안된다.
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        # 유닛들이 수행할 액션은 리스트 형태로 만들어서,
        # do_actions 함수에 인자로 전달하면 게임에서 실행된다.
        # do_action 보다, do_actions로 여러 액션을 동시에 전달하는
        # 것이 훨씬 빠르다.
        actions = list()
        self.cached_known_enemy_units = self.known_enemy_units()
        self.cached_known_enemy_structures = self.known_enemy_structures()
        actions += await self.unit_actions()

        await self.do_actions(actions)

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def select_threat(self, unit:Unit):
        # 자신에게 위협이 될 만한 상대 유닛들을 리턴
        threats = None
        if unit.is_flying:
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

    def select_mode(self, unit:Unit):
        # 방어모드일때 공격모드로 전환될지 트리거 세팅
        nearby_enemies = self.known_enemy_units.filter(
            lambda u: unit.distance_to(u) <= max(unit.sight_range, unit.ground_range, unit.air_range))
        if nearby_enemies.empty:
            self.evoked[(unit.tag, "offense_mode")] = False
        else:
            self.evoked[(unit.tag, "offense_mode")] = True

    # 무빙샷
    def moving_shot(self, actions, unit, cooldown, target_func, margin_health:float=0, minimum:float=0):
        # print("WEAPON COOLDOWN : ", unit.weapon_cooldown)
        if unit.weapon_cooldown < cooldown:
            target = target_func(unit)
            if self.time - self.evoked.get((unit.tag, "COOLDOWN"), 0.0) >= minimum:
                actions.append(unit.attack(target))
                self.evoked[(unit.tag, "COOLDOWN")] = self.time

        elif (margin_health == 0 or unit.health_percentage <= margin_health) and self.time - self.evoked.get(
                (unit.tag, "COOLDOWN"), 0.0) >= minimum:  # 무빙을 해야한다면
            # print("GOAL")
            maxrange = 0
            total_move_vector = Point2((0, 0))
            showing_only_enemy_units = self.known_enemy_units.not_structure.filter(lambda e: e.is_visible)

            if not unit.is_flying:
                # 배틀크루저 예외처리.
                threats = showing_only_enemy_units.filter(
                    lambda u: (u.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(u)) or (
                                u.can_attack_ground and u.ground_range + 2 >= unit.distance_to(u)))
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
                    lambda u: (u.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(u)) or (
                            u.can_attack_air and u.air_range + 2 >= unit.distance_to(u)))
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
        center_candidates = self.units.not_structure.filter(lambda u:u.type_id is not UnitTypeId.SIEGETANKSIEGED and u.type_id is not UnitTypeId.SIEGETANK)
        for unit in center_candidates:
            group = center_candidates.closer_than(5, unit)
            groups.append(group)

        groups.sort(key=lambda g: g.amount, reverse=True)
        my_groups = []

        # groups가 비는 경우는 시즈탱크 제외 유닛이 아예 없다는 것
        # 이 경우 빈 list 반환
        if not groups:
            return my_groups

        # groups가 비지 않는 경우
        my_groups.append(groups[0])
        selected_units = groups[0]

        group_num = int(self.units.not_structure.amount / 10.0)

        for i in range(0, group_num):
            groups.sort(key=lambda g: (g - selected_units).amount, reverse=True)
            # groups.sorted(lambda g : g.filter(lambda u : not (u in selected_units)), reverse=True)
            my_groups.append(groups[0])
            selected_units = selected_units or groups[0]

        return my_groups

    async def unit_actions(self):
        #
        # 유닛 명령 생성
        #
        actions = list()

        if not self.units.not_structure.empty:
            my_groups = self.unit_groups()

        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택

            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(self.enemy_cc) < unit.distance_to(enemy_unit):
                target = self.enemy_cc
            else:
                target = enemy_unit

            # 모든 유닛에 대해 방어중일 때 공격형 모드로 전환될지 말지 설정
            # 설정은 하나 화염차는 예외로 사용은 하지 않는다.
            self.select_mode(unit)

            ## 의료선과 밤까마귀 아니면 ...
            if unit.type_id is not (UnitTypeId.MEDIVAC and UnitTypeId.RAVEN):
                if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                    pass
                    # 전투가능한 유닛 수가 15를 넘으면 적 본진으로 공격
                    # actions.append(unit.attack(target))
                # else:  # ArmyStrategy.DEFENSE
                # 적 사령부 방향에 유닛 집결
                # target = self.start_location + 0.25 * (self.enemy_cc.position - self.start_location)
                # actions.append(unit.attack(target))

                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if (self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False)) and unit.distance_to(target) < 8:
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
                                return enemies.sorted(lambda u:u.health)[0]
                                #return enemies.closest_to(unit)

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
                            threats = self.known_enemy_units.filter(lambda u: u.can_attack_air and u.air_range + 2 >= unit.distance_to(u))
                            for eunit in self.known_enemy_units:
                                if eunit.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(eunit):
                                    threats.append(eunit)

                            if threats.empty:
                                if self.known_enemy_units.empty:
                                    return self.enemy_cc
                                else:
                                    return self.known_enemy_units.closest_to(unit)  # 가까운애 때리기
                            else:
                                return threats.sorted(lambda u:u.health)[0]

                        def yamato_target_func(unit):
                            # 야마토 포 상대 지정
                            # 일정 범위 내 적들에 한해 적용
                            yamato_enemy_range = 15
                            yamato_candidate_id = [UnitTypeId.THORAP, UnitTypeId.THOR, UnitTypeId.BATTLECRUISER, UnitTypeId.SIEGETANKSIEGED,
                                                   UnitTypeId.SIEGETANK, UnitTypeId.RAVEN]

                            for eunit_id in yamato_candidate_id:
                                target_candidate = self.known_enemy_units.filter(lambda u: u.type_id is eunit_id and unit.distance_to(u) <= yamato_enemy_range)
                                target_candidate.sorted(lambda u: u.health, reverse=True)
                                if not target_candidate.empty:
                                    return target_candidate.first

                            # 위 리스트 안 개체들이 없다면 나머지 중 타겟팅
                            # 나머지 유닛도 없다면 적 커맨드로 ㄱㄱ
                            enemy_left = self.known_enemy_units.filter(lambda u: unit.distance_to(u) <= yamato_enemy_range)
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
                                lambda u: u.is_light) < flying_enemies.filter(lambda u: u.is_armored)):  # 경장갑이 적으면
                            actions.append(unit(AbilityId.MORPH_THORHIGHIMPACTMODE))  # 250mm 천벌포로 교체
                            self.evoked[(unit.tag, "CHANGE_WEAPON")] = self.time

                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):  # 재블린 미사일 모드
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
                                lambda u: u.is_light) > flying_enemies.filter(lambda u: u.is_armored)):  # 경장갑이 많으면
                            actions.append(unit(AbilityId.MORPH_THOREXPLOSIVEMODE))  # 재블린 모드로 교체
                            self.evoked[(unit.tag, "CHANGE_WEAPON")] = self.time

                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):  # 250mm 천벌포 모드
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
                    print("mode : ", self.evoked.get((unit.tag, "offense_mode"), False))
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        # 먼저 타겟부터 정하자.
                        # 적이 보이고 사정거리 내에 있는데 붙어 있지 않으면 바로 시즈모드
                        if self.cached_known_enemy_units.exists:
                            # 시즈탱크가 공격 가능한 지상 유닛이어야 함
                            enemy_ground_units = self.cached_known_enemy_units.filter(lambda u: not u.is_flying)
                            mindist = 500  # mindist
                            closest_pos = Point2()
                            for eunit in enemy_ground_units:
                                # 각 적 유닛간의 거리 조사
                                if mindist > unit.distance_to(eunit):
                                    mindist = unit.distance_to(eunit)
                                    closest_pos = eunit.position
                            # 너무 멀다면 전진배치.
                            # 그룹 뒤에 탱크를 놓는 것은 12/31일 논의.

                            ## add add add

                            # groups 쓰면 댐

                            threats = self.cached_known_enemy_units.filter(
                                lambda u: u.can_attack_ground and u.ground_range + 2 >= unit.distance_to(u))

                            if mindist > 13.0 and mindist < 500:
                                actions.append(unit.attack(closest_pos))
                            # 적정 거리에 들어왔다면 시즈모드.
                            elif mindist > 9.0 and mindist <= 13.0 and threats.empty:
                                actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))

                            # 적이 날 공격하려 한다면 반대방향으로 튀자..
                            # 무빙샷 구현

                            def target_func(unit):

                                selected_enemies = self.known_enemy_units.filter(
                                    lambda u: u.is_visible and not eunit.is_flying)
                                if selected_enemies.empty:
                                    return self.enemy_cc
                                else:
                                    return selected_enemies.closest_to(unit)

                            actions = self.moving_shot(actions, unit, 3, target_func)

                        # 보이는 게 하나도 없으면 커맨드로 달려간다.
                        else:
                            actions.append(unit.attack(self.enemy_cc))

                    # 설정된 정책이 방어이고 근처에 적이 없는 경우를 의미
                    # 그룹 센터에서 상대적으로 뒤쪽에 대기한다.
                    # 그룹 센터에서 거리는 왼쪽으로 랜덤으로 정해지되, 5-9 정도.
                    # 근처에 적이 없을 때 positioning, 아니면 offense 상태로 이행
                    else:
                        # 포지셔닝
                        # 예외 처리 : 만약 그룹 센터가 아예 없는 경우는 아무것도 하지 않음.
                        if my_groups:
                            # 만약 첫 프레임이거나 이전 프레임에 설정된 그룹 센터와 현재 계산된 그룹 센터가 일정 거리 이상(3)다르다면 이동1
                            if self.evoked.get((unit.tag, "desire_add_vector"), None) is None:
                                print("1")
                                dist = random.randint(5, 9)
                                dist_x = random.randint(2, dist)
                                dist_y = math.sqrt(dist ** 2 - dist_x ** 2) if random.randint(0, 1) == 0 else -math.sqrt(
                                    dist ** 2 - dist_x ** 2)
                                desire_add_vector = Point2((-dist_x, dist_y))
                                desired_pos = my_groups[0].center + desire_add_vector
                                desired_pos = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                               self.clamp(desired_pos.y, 0, self.map_height)))
                                self.evoked[("group_center")] = my_groups[0].center
                                self.evoked[(unit.tag, "desire_add_vector")] = desire_add_vector
                                actions.append(unit.move(desired_pos))
                            else:
                                print("2")
                                if my_groups[0].center.distance_to(self.evoked.get(("group_center"), None)) > 3:
                                    self.evoked[("group_center")] = my_groups[0].center
                                desired_pos = self.evoked.get(("group_center"), None) + self.evoked.get((unit.tag, "desire_add_vector"), None)
                                if int(unit.position.x) == int(desired_pos.x) and int(unit.position.y) == int(desired_pos.y):
                                    actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                                else:
                                    actions.append(unit.move(desired_pos))


                # 시즈모드 시탱
                if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        if self.cached_known_enemy_units.exists:
                            # 타겟팅 정하기
                            targets = self.cached_known_enemy_units.in_attack_range_of(unit)
                            min_HP = 1e9
                            for eunit in targets:
                                if eunit.is_visible and not eunit.is_flying and eunit.health < min_HP:
                                    # 중장갑 위주 + HP 적은놈
                                    if target is not None and target.is_armored and not eunit.is_armored:
                                        continue
                                    target = eunit
                                    min_HP = eunit.health
                            if target is not None:
                                actions.append(unit.attack(target))

                            # 시즈모드 풀지 안풀지 결정하기
                            # 나와있는 ground_range에 0.7쯤 더해야 실제 사정거리가 된다..
                            # 넉넉잡아 2로..
                            threats = self.cached_known_enemy_units.filter(
                                lambda u: u.can_attack_ground and u.ground_range + 2 >= unit.distance_to(u))
                            # 한 유닛이라도 자신을 때릴 수 있으면 바로 시즈모드 해제
                            if threats.amount > 0:
                                actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))

                        # 보이는 애들이 없으면 다시 시즈모드를 푼다.
                        else:
                            actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                    # 만약 방어적일때 그룹 센터가 일정 거리 이상(3)달라지면 시즈모드 풀기(이동 준비)
                    elif my_groups:
                        if my_groups[0].center.distance_to(self.evoked.get(("group_center"), None)) > 3:
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
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        def target_func(unit):
                            target = None
                            min_dist = math.sqrt(self.map_height ** 2 + self.map_width ** 2) + 10
                            for eunit in self.cached_known_enemy_units:
                                if eunit.is_visible and not eunit.is_flying and eunit.distance_to(unit) < min_dist:
                                    # 경장갑 우선 타겟팅
                                    if target is not None and target.is_light and not eunit.is_light:
                                        continue
                                    target = eunit
                                    min_dist = eunit.distance_to(unit)
                            if target is None:
                                enemies = self.known_enemy_units.filter(lambda e: e.is_visible)

                                if not enemies.empty:
                                    target = enemies.closest_to(unit)
                                else:
                                    target = self.enemy_cc
                            return target

                        actions = self.moving_shot(actions, unit, 10, target_func)
                    else:  # 정찰모드
                        # evoke initialize

                        if (unit.tag, "scout") not in self.evoked:
                            self.evoked[(unit.tag, "scout")] = False

                        if self.evoked.get((unit.tag, "scout"), False):
                            # 아무것도 하지 않을 때
                            # 목표 지점에 도달한 것도 포함
                            if unit.is_idle:
                                self.evoked[(unit.tag, "scout")] = False
                            # 적을 만났을 때
                            for eunit in self.cached_known_enemy_units:
                                if unit.distance_to(eunit) < unit.sight_range:
                                    self.evoked[(unit.tag, "scout")] = False
                                    break

                        # 정찰 중이 아닐 때 정찰나감
                        if not self.evoked.get((unit.tag, "scout"), False):
                            dest = None
                            if not unit.is_moving and not unit.is_attacking:
                                if self.is_visible(self.enemy_cc):
                                    dest = Point2(
                                        (random.randint(0, self.map_width), random.randint(0, self.map_height)))
                                    actions.append(unit.attack(dest))
                                else:
                                    # 커맨드 invisible이면 우선 보이게 하는 것이 목적
                                    dest = self.enemy_cc
                                    actions.append(unit.attack(dest))
                                self.evoked[(unit.tag, "scout")] = True
                            else:
                                # 정찰 중이 아닌데, 움직이고 있거나 공격 중일때 발생
                                # 쿨타임 중에는 빠지기
                                # 쿨타임이 차면 공격한 다음 빠지기
                                # 단위가 frame
                                def target_func(unit):
                                    target = None
                                    min_dist = math.sqrt(self.map_height ** 2 + self.map_width ** 2) + 10
                                    for eunit in self.cached_known_enemy_units:
                                        if eunit.is_visible and not eunit.is_flying and eunit.distance_to(
                                                unit) < min_dist:
                                            # 경장갑 우선 타겟팅
                                            if target is not None and target.is_light and not eunit.is_light:
                                                continue
                                            target = eunit
                                            min_dist = eunit.distance_to(unit)
                                    return target

                                actions = self.moving_shot(actions, unit, 10, target_func)

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
                        if not self.known_enemy_units.filter(lambda e: e.is_flying).empty:
                            def target_func(unit):
                                target = \
                                self.known_enemy_units.filter(lambda e: e.is_flying).sorted(lambda e: e.health)[0]
                                return target

                            actions = self.moving_shot(actions, unit, 15, target_func)

                        # 우선순위 2
                        else:
                            # 우선순위 2로 이행
                            sigeged_targets = self.cached_known_enemy_units.filter(
                                lambda u: u.type_id is UnitTypeId.SIEGETANKSIEGED)
                            tank_targets = self.known_enemy_units.filter(lambda u: u.type_id is UnitTypeId.SIEGETANK)

                            if sigeged_targets.amount > 0:
                                target = sigeged_targets.sorted(lambda e: e.health)[0]
                                actions.append(unit.move(target))

                                if unit.distance_to(target) <= 5.0:
                                    actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))
                                    actions.append(unit.attack(target))

                            elif tank_targets.amount > 0:
                                target = tank_targets.sorted(lambda e: e.health)[0]
                                actions.append(unit.move(target))

                                if unit.distance_to(target) <= 5.0:
                                    actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))
                                    actions.append(unit.attack(target))
                            else:
                                # 탱크가 없으므로 보이는 지상 유닛이 있다면 가장 HP가 없는 걸 때리기
                                targets = self.cached_known_enemy_units.filter(
                                    lambda u: unit.sight_range > unit.distance_to(u))
                                if not targets.empty:
                                    target = targets.sorted(lambda e: e.health)[0]

                                    actions.append(unit.move(target))

                                    if unit.distance_to(target) <= 5.0:
                                        actions.append(unit(AbilityId.MORPH_VIKINGASSAULTMODE))
                                        actions.append(unit.attack(target))

                                else:
                                    # 우선순위 3으로 이행
                                    # 보이는 공중 유닛, 지상 유닛 아무도 없음
                                    # 커맨드를 때리러 간다.
                                    actions.append(unit.attack(self.enemy_cc))

                # 바이킹 전투 모드(지상)
                if unit.type_id is UnitTypeId.VIKINGASSAULT:
                    if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                        # 탱크가 가까이 있으면 공격
                        enemy_tanks = self.known_enemy_units.filter(
                            lambda u: u.type_id is (UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                        )

                        if enemy_tanks.filter(lambda e: unit.distance_to(e) < 6.0).amount > 0:
                            actions.append(unit.attack(enemy_tanks.sorted(lambda e: e.health)[0]))

                        # 반대로 탱크가 멀리 있고 사정거리 내에 있다면?
                        # 탱크의 사정거리 내에 있으면(시즈 모드 거리에) 전투기 변환으로 공격 회피
                        elif enemy_tanks.filter(lambda e: unit.distance_to(e) < e.ground_range + 2).amount > 0:
                            actions.append(unit(AbilityId.MORPH_VIKINGFIGHTERMODE))


                        else:
                            # 주위에 공중 유닛이 있다면 전투기로 변환하여 공격
                            targets = self.cached_known_enemy_units.filter(
                                lambda u: u.is_flying and unit.sight_range > unit.distance_to(u))
                            if not targets.empty:
                                actions.append(unit(AbilityId.MORPH_VIKINGFIGHTERMODE))

                            else:
                                # 아무도 없으면 커맨드, 사정거리 내 지상유닛 누군가가 보인다면 그놈을 조지자
                                ground_units = self.cached_known_enemy_units.not_flying.filter(lambda e: e.is_visible)
                                if ground_units.empty:
                                    actions.append(unit(AbilityId.MORPH_VIKINGFIGHTERMODE))
                                else:
                                    def target_func(unit):
                                        ground_units = self.cached_known_enemy_units.not_flying.filter(
                                            lambda e: e.is_visible)

                                        if not ground_units.empty:
                                            return ground_units.closest_to(unit)

                                        return self.enemy_cc

                                    actions = self.moving_shot(actions, unit, 1, target_func, 0.5)

                '''
                ## BANSHEE ##

                if unit.type_id is UnitTypeId.BANSHEE:
                    if self.army_strategy is ArmyStrategy.OFFENSE and unit.distance_to(target) < 15:
                        if not unit.has_buff(BuffId.BANSHEECLOAK) and unit.energy > 50 : # 은폐 아닌데 마나 50 넘으면 은폐 하기
                            actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_BANSHEE))

                        for enemy in self.known_enemy_units :
                            if enemy.type_id is UnitTypeId.SIEGETANK : # 만약 시즈탱크가 있으면
                                actions.append(unit.attack(enemy)) # 그거 때리기

                ## BANSHEE END ##
                '''
                ## REAPER ##

                if unit.type_id is UnitTypeId.REAPER and self.army_strategy is ArmyStrategy.OFFENSE:
                    if unit.health_percentage <= .3:  # 30퍼 이하면 도망
                        actions.append(unit.move(self.start_location))
                        self.evoked[(unit.tag, "REAPER_RUNAWAY")] = True
                    if self.evoked.get((unit.tag, "REAPER_RUNAWAY"), False):
                        if unit.health_percentage >= 0.7:  # 70퍼 이상이면 다시 전투 진입
                            self.evoked[(unit.tag, "REAPER_RUNAWAY")] = False
                    if not self.evoked.get((unit.tag, "REAPER_RUNAWAY"), False):
                        actions.append(unit.attack(target))
                        # 가까운 적과 거리 비례로 도망가기

                ## REAPER END

                ## wonseok end ##

            if unit.type_id is UnitTypeId.MEDIVAC:
                if self.wounded_units.exists:
                    wounded_unit = self.wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    if self.combat_units.exists:
                        actions.append(unit.move(my_groups[0].center))

            ### wonseok add ###

            ## RAVEN ##

            if unit.type_id is UnitTypeId.RAVEN:
                if unit.distance_to(
                        target) < 15 and unit.energy > 75 and self.army_strategy is ArmyStrategy.OFFENSE:  # 적들이 근처에 있고 마나도 있으면
                    known_only_enemy_units = self.known_enemy_units.not_structure
                    if known_only_enemy_units.exists:  # 보이는 적이 있다면
                        enemy_amount = self.known_only_enemy_units.amount
                        not_antiarmor_enemy = self.known_only_enemy_units.filter(
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
                        actions.append(unit.move(my_groups[0].center))

                ## RAVEN END ##

            ## GHOST ##

            if unit.type_id is UnitTypeId.GHOST:
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
                                print(select_unit)
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
                            enemy_biological = self.known_enemy_units.filter(lambda e: e.is_biological)
                            if not enemy_biological.empty:
                                target = enemy_biological.closest_to(unit)  # 가장 가까운 생체유닛 저격
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

                check = False

                # 무언가 목표물이 있어서 이동하거나 공격 중이거나 유휴 상태
                # 목표물을 찾아, 치고 빠지기 구현

                def target_func(unit):
                    check = False
                    # 시즈탱크
                    if not check:
                        query_units = self.cached_known_enemy_units.filter(lambda
                                                                               u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED).sorted(
                            lambda u: u.health + unit.distance_to(u))
                        if not query_units.empty:
                            return query_units.first
                    # 토르
                    if not check:
                        query_units = self.cached_known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.THOR or u.type_id is UnitTypeId.THORAP).sorted(
                            lambda u: u.health + unit.distance_to(u))
                        if not query_units.empty:
                            return query_units.first
                    # 가까운거
                    if not check:
                        query_units = self.known_enemy_units.filter(lambda e: e.is_visible)
                        if not query_units.empty:
                            return query_units.closest_to(unit)

                    # 커맨드
                    if not check:
                        actions.append(unit.attack(self.enemy_cc))

                actions = self.moving_shot(actions, unit, 3, target_func, 0.5)

            # 밴시
            # 공성전차를 위주로 잡게 한다.
            # 기본적으로 공성전차를 찾아다니되 들키면 튄다 ㅎㅎ
            if unit.type_id is UnitTypeId.BANSHEE:
                if self.army_strategy is ArmyStrategy.OFFENSE or self.evoked.get((unit.tag, "offense_mode"), False):
                    def target_func(unit):
                        enemy_tanks = self.cached_known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                        if enemy_tanks.amount > 0:
                            target = enemy_tanks.closest_to(unit.position)
                            return target
                        # 만약 탱크가 없다면 HP가 가장 적은 아무 지상 유닛이나, 그것도 없다면 커맨드 직행
                        else:
                            targets = self.cached_known_enemy_units.not_flying

                            if not targets.empty:
                                target = targets.sorted(lambda e: e.health)[0]
                                return target

                            targets = self.known_enemy_units.filter(lambda e: e.is_visible)

                            if not targets.empty:
                                target = targets.closest_to(unit)
                                return target

                            return self.enemy_cc

                    # 은신 쓰기
                    threats = self.cached_known_enemy_units.filter(
                        lambda u: u.can_attack_air and u.air_range + 2 >= unit.distance_to(u))

                    if (not unit.has_buff(BuffId.BANSHEECLOAK) and unit.energy_percentage >= 0.3) and not threats.empty:
                        actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_BANSHEE))

                    # 공격
                    # 은신 상태이면서 밤까마귀가 감지하고 있지 않으면 그냥 공격
                    # 그 상태가 아니라면 무빙샷.
                    # TODO : 은신이 감지되고 있는지 확인이 불가함. CloakState 작동 불가
                    # 불가피하게 밤까마귀의 탐지 범위 안에 있는지 확인하는 것으로 대체
                    clock_threats = self.cached_known_enemy_units.filter(
                        lambda u: u.type_id is UnitTypeId.RAVEN and unit.distance_to(u) <= u.detect_range)

                    # 공격 모드 전환
                    if unit.is_idle and unit.energy_percentage >= 0.3:
                        actions.append(unit.attack(target_func(unit)))

                    # 이미 교전 중일때 전략 결정
                    if not unit.is_idle:
                        if unit.has_buff(BuffId.BANSHEECLOAK) and clock_threats.empty:
                            actions.append(unit.attack(target_func(unit)))
                        else:
                            actions = self.moving_shot(actions, unit, 5, target_func)

                    # 만약 주위에 아무도 자길 때릴 수 없으면 클락을 풀어 마나보충
                    if not threats.empty:
                        self.evoked[(unit.tag, "BANSHEE_CLOAK")] = self.time

                    if threats.empty and (
                            self.time - self.evoked.get((unit.tag, "BANSHEE_CLOAK"), 0.0) >= 10) and unit.has_buff(
                            BuffId.BANSHEECLOAK):
                        actions.append(unit(AbilityId.BEHAVIOR_CLOAKOFF_BANSHEE))

            # 지게로봇
            # 에너지 50을 사용하여 소환
            # 일정 시간 뒤에 파괴된다.
            # 용도 : 사령부 수리 or 메카닉 유닛 수리

            if unit.type_id is UnitTypeId.MULE:
                if self.cc.health < self.cc.health_max:
                    # 커맨드 수리
                    actions.append(unit(AbilityId.EFFECT_REPAIR_MULE , self.cc))
                else:
                    # 근처 수리 가능한 메카닉 애들을 찾아 수리
                    # 아마 커맨드 근처에 있는 애들이 될 것임.
                    # 어차피 일정 시간 뒤 파괴되므로 HP가 가장 적은 애들을 찾는 것보다는 근처 애들이 나음
                    repair_candidate = self.units.not_structure.filter(lambda u: u.is_mechanical and u.health < u.health_max)
                    if not repair_candidate.empty:
                        repair_target = repair_candidate.closest_to(unit)
                        actions.append(unit(AbilityId.EFFECT_REPAIR_MULE , repair_target))


        return actions