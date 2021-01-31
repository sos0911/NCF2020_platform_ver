__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time
import numpy as np
import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2

from sc2.data import Result
from sc2.data import CloakState
from sc2.player import Bot as _Bot
from termcolor import colored, cprint
from sc2.pixel_map import PixelMap
import random
import math


class Bot(sc2.BotAI):
    """
    해병 5, 의료선 1 빌드오더를 계속 실행하는 봇
    해병은 적 사령부와 유닛중 가까운 목표를 향해 각자 이동
    적 유닛또는 사령부까지 거리가 15미만이 될 경우 스팀팩 사용
    스팀팩은 체력이 50% 이상일 때만 사용가능
    의료선은 가장 가까운 체력이 100% 미만인 해병을 치료함
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build_order = list() # 생산할 유닛 목록

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.build_order = list()
        self.evoked = dict()
        self.my_raven = False
        self.maybe_battle = False

        self.map_height = 63
        self.map_width = 128

        self.train_raven = False
        self.attacking = False
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first

        self.my_groups = []

        # (32.5, 31.5) or (95.5, 31.5)
        if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0:
            self.enemy_cc = Point2(Point2((95.5, 31.5)))  # 적 시작 위치
            self.rally_point = Point2(Point2((47.5, 31.5)))
            self.escape_point = Point2(Point2((30.5, 31.5)))
        else:
            self.enemy_cc = Point2(Point2((32.5, 31.5)))  # 적 시작 위치
            self.rally_point = Point2(Point2((80.5, 31.5)))
            self.escape_point = Point2(Point2((97.5, 31.5)))

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

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
                    threats = showing_only_enemy_units.filter(
                        lambda u: ((u.type_id is UnitTypeId.BATTLECRUISER and 6 + 2 >= unit.distance_to(u)) or (
                                u.can_attack_ground and u.ground_range + 3 >= unit.distance_to(u))))
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
                    threats = showing_only_enemy_units.filter(
                        lambda u: ((u.type_id is UnitTypeId.BATTLECRUISER and 6 + 3 >= unit.distance_to(u)) or (
                                u.can_attack_air and u.air_range + 3 >= unit.distance_to(u))))
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

    def select_threat(self, unit):
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

    def unit_groups(self):
        groups = []
        center_candidates = self.units.not_structure.filter(
            lambda u: u.type_id is not UnitTypeId.SIEGETANKSIEGED and u.type_id is not UnitTypeId.SIEGETANK)
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

    def select_mode(self, unit):
        # 정찰중인 벌쳐는 제외

        # 밴시, 밤까마귀, 지게로봇 제외
        if unit.type_id is UnitTypeId.MULE or unit.type_id is UnitTypeId.RAVEN :
            #self.evoked[(unit.tag, "offense_mode")] = False
            return None
        # 방어모드일때 공격모드로 전환될지 트리거 세팅
        # 방어모드라면 False, 공격모드로 바뀐다면 True return

        ground_targets = self.known_enemy_units.filter(
                lambda u: unit.distance_to(u) <= max(unit.ground_range + 2, unit.air_range + 2) and u.is_visible
                    and not u.is_flying)
        air_targets = self.known_enemy_units.filter(
                lambda u: unit.distance_to(u) <= max(unit.ground_range + 2, unit.air_range + 2) and u.is_visible
                    and u.is_flying)

        if ground_targets.exists and air_targets.exists :
            return "All"
        elif ground_targets.exists :
            return "Ground"
        elif air_targets.exists :
            return "Air"
        else :
            return None

    async def on_step(self, iteration: int):       
        actions = list()

        if not self.units.not_structure.empty:
            self.my_groups = self.unit_groups()
        #
        # 빌드 오더 생성
        #
        if self.attacking == True and self.units(UnitTypeId.BANSHEE).amount <= 2 :
            self.evoked['go'] = False


        if self.evoked.get('go', False) :
            self.attacking = True
        elif self.units(UnitTypeId.BANSHEE).amount >= 6 :
            if not self.evoked.get('ready', False) :
                self.evoked['ready'] = True
                self.evoked['ready_time'] = self.time
            elif self.time - self.evoked['ready_time'] >= 6.0 : 
                self.evoked['go'] = True
                self.attacking = True
        #elif self.known_enemy_units.not_structure.exists :
        #    for unit in self.units :
        #        if(self.known_enemy_units.not_structure.filter(lambda e : e.is_visible).in_attack_range_of(unit).exists) :
        #            self.attacking = True
        #            break
        #        self.attacking = False
        else :
            self.attacking = False

        self.train_raven = False
        for enemy in self.known_enemy_units :
            if enemy.type_id is UnitTypeId.BANSHEE and self.units(UnitTypeId.RAVEN).empty:
                self.train_raven = True
                break

        if self.time - self.evoked.get('create', 0) > 1.0 and self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0: 
            if self.train_raven and not UnitTypeId.RAVEN in self.build_order and self.units(UnitTypeId.RAVEN).empty: # 상대한테 벤시가 있고 베스핀 175 이상이고 레이븐을 추가한 상태가 아니고 레이븐이 없어야함
                if self.vespene > 175 :
                    self.build_order = [UnitTypeId.RAVEN]
                    self.evoked['create'] = self.time
                elif not self.build_order :
                    self.build_order.insert(0, UnitTypeId.MARINE)
                    self.evoked['create'] = self.time

            elif self.vespene > 60 and not UnitTypeId.BANSHEE in self.build_order and not UnitTypeId.RAVEN in self.build_order:
                self.build_order.append(UnitTypeId.BANSHEE)
                self.evoked['create'] = self.time

            elif self.vespene <= 60 and not self.build_order:
                self.build_order.insert(0, UnitTypeId.MARINE)
                self.evoked['create'] = self.time

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 왠지는 모르겠는데 이걸 추가해야 실시간 tracking이 된다..

        #
        # 사령부 명령 생성
        #
        if self.build_order and self.can_afford(self.build_order[0]) and self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
            # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
            actions.append(self.cc.train(self.build_order[0]))  # 첫 번째 유닛 생산 명령 
            if self.build_order[0] == UnitTypeId.RAVEN :
                self.train_raven = False
            del self.build_order[0]  # 빌드오더에서 첫 번째 유닛 제거
            self.evoked[(self.cc.tag, 'train')] = self.time

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

        '''
        closest_dist = 500
        enemies = self.known_enemy_units.filter(lambda e: e.is_visible)
        if not enemies.empty:
            for our_unit in self.units:
                if our_unit.type_id is UnitTypeId.BANSHEE :
                    continue
                temp = enemies.closest_distance_to(our_unit)
                if temp < closest_dist:
                    closest_dist = temp
        '''
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

        if ground_attack and air_attack:
            self.offense_mode = True
            for unit in self.units.not_structure:
                if self.evoked.get(("scout_unit_tag"), None) is not None and unit.tag == self.evoked.get(
                        ("scout_unit_tag")):
                    self.evoked[(unit.tag, "offense_mode")] = False
                else:
                    self.evoked[(unit.tag, "offense_mode")] = True

        elif ground_attack:
            self.offense_mode = True
            for unit in self.units.not_structure.filter(
                    lambda u: u.can_attack_ground or u.type_id in [UnitTypeId.RAVEN, UnitTypeId.MEDIVAC,
                                                                   UnitTypeId.VIKINGFIGHTER, \
                                                                   UnitTypeId.BATTLECRUISER]):
                if self.evoked.get(("scout_unit_tag"), None) is not None and unit.tag == self.evoked.get(
                        ("scout_unit_tag")):
                    self.evoked[(unit.tag, "offense_mode")] = False
                else:
                    self.evoked[(unit.tag, "offense_mode")] = True

        elif air_attack:
            self.offense_mode = True
            for unit in self.units.not_structure.filter(
                    lambda u: u.can_attack_air or u.type_id in [UnitTypeId.RAVEN, UnitTypeId.MEDIVAC,
                                                                UnitTypeId.VIKINGASSAULT, \
                                                                UnitTypeId.BATTLECRUISER]):
                self.evoked[(unit.tag, "offense_mode")] = True
        else:
            for unit in self.units.not_structure:
                self.evoked[(unit.tag, "offense_mode")] = False
                self.offense_mode = False

        # 1등 코드가, 밤까마귀로 하여금 무조건 클라킹한 밴시만 따라다니도록 한다;;
        enemy_banshees = self.known_enemy_units.filter(lambda u: u.type_id is UnitTypeId.BANSHEE)

        for unit in self.units.not_structure :

            enemy_units = self.known_enemy_units.filter(lambda u: u.is_visible)
            if enemy_units.in_attack_range_of(unit).exists:
                target = enemy_units.in_attack_range_of(unit).closest_to(unit)  # 가장 가까운 적 유닛
            else:
                target = self.enemy_cc

            if not unit.type_id in [UnitTypeId.RAVEN, UnitTypeId.MULE]:
                #if unit.type_id is UnitTypeId.RAVEN:
                #    pass
                #    #print("check")
                is_attacked = False
                visible_enemy = self.known_enemy_units.not_structure.filter(lambda e : e.is_visible)
                for e in visible_enemy :
                    if e.target_in_range(unit) :
                        is_attacked = True
                        break

                if self.attacking == False and (not self.evoked.get((unit.tag, "offense_mode"), False) and not is_attacked):

                    if unit.type_id is UnitTypeId.MARINE and self.known_enemy_units(UnitTypeId.BANSHEE).filter(lambda e : not e.is_visible) :
                        actions.append(unit.attack(self.escape_point))
                    else :   
                        actions.append(unit.attack(self.rally_point))

            if unit.type_id is UnitTypeId.MARINE :
                if (self.attacking == True or self.evoked.get((unit.tag, "offense_mode"), False)) :
                    actions.append(unit.attack(target))
                    if unit.distance_to(target) < 15 and self.known_enemy_units.amount >= 3:
                        # 해병과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time
                

            # 밴시
            # 공성전차를 위주로 잡게 한다.
            # 기본적으로 공성전차를 찾아다니되 들키면 튄다 ㅎㅎ
            if unit.type_id is UnitTypeId.BANSHEE:

                # 아래 내용은 공격 정책이거나 offense mode가 아닐 시에도 항시 적용됨
                threats = self.select_threat(unit)
                # clock_threats = self.cached_known_enemy_units.filter(
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

                # attacking 시에
                if (self.attacking == True or self.evoked.get((unit.tag, "offense_mode"), False)):

                    def target_func(unit):
                        enemy_tanks = self.known_enemy_units.filter(
                            lambda u: u.type_id is UnitTypeId.SIEGETANK or u.type_id is UnitTypeId.SIEGETANKSIEGED)
                        if enemy_tanks.amount > 0:
                            target = enemy_tanks.closest_to(unit.position)
                            return target
                        # 만약 탱크가 없다면 HP가 가장 적으면서 가까운 아무 지상 유닛이나, 그것도 없다면 커맨드 직행
                        else:
                            targets = self.known_enemy_units.filter(
                                lambda u: u.type_id is not UnitTypeId.COMMANDCENTER and not u.is_flying)
                            max_dist = math.sqrt(self.map_height ** 2 + self.map_width ** 2)
                            if not targets.empty:
                                target = targets.sorted(lambda u: u.health_percentage + unit.distance_to(u) / max_dist)[0]
                                return target
                            else:
                                return self.enemy_cc

                    # 공격
                    # 은신 상태이면서 밤까마귀가 감지하고 있지 않으면 그냥 공격
                    # 그 상태가 아니라면 무빙샷.
                    # TODO : 은신이 감지되고 있는지 확인이 불가함. CloakState 작동 불가
                    # 무빙샷 함수 안에 cloak에 대한 예외처리도 되어 있다.
                    actions = self.moving_shot(actions, unit, 5, target_func)

            # RAVEN
            if unit.type_id is UnitTypeId.RAVEN:

                threats = self.select_threat(unit)  # 위협이 있으면 ㅌㅌ
                banshees = self.known_enemy_units(UnitTypeId.BANSHEE).closer_than(unit.sight_range, unit)
                our_auto_turrets = self.units(UnitTypeId.AUTOTURRET)

                if not (self.attacking == True or self.evoked.get((unit.tag, "offense_mode"), False)) \
                        and banshees.exists and unit.energy > 50 and threats.empty:
                    if our_auto_turrets.empty or (
                            not our_auto_turrets.empty and our_auto_turrets.closest_distance_to(unit) < 10):
                        build_loc = banshees.center
                        if await self.can_place(building=AbilityId.BUILDAUTOTURRET_AUTOTURRET,
                                                position=build_loc):
                            actions.append(unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, build_loc))

                elif unit.distance_to(target) < 15 and unit.energy > 75 and \
                        (self.attacking == True or self.evoked.get((unit.tag, "offense_mode"), False)):  # 적들이 근처에 있고 마나도 있으면
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

            # 지게로봇
            # 에너지 50을 사용하여 소환
            # 일정 시간 뒤에 파괴된다.
            # 용도 : 사령부 수리 or 메카닉 유닛 수리

            if unit.type_id is UnitTypeId.MULE:
                if self.cc.health < self.cc.health_max:
                    # 커맨드 수리
                    actions.append(unit(AbilityId.EFFECT_REPAIR_MULE, self.cc))
                else:
                    # 할게 없는 상태.
                    # 평소 대기 시에는 우리 커맨드보다 조금 안쪽에서 대기
                    if self.cc.position.x < 50:
                        actions.append(unit.move(Point2((self.cc.position.x - 5, self.cc.position.y))))
                    else:
                        actions.append(unit.move(Point2((self.cc.position.x + 5, self.cc.position.y))))

        await self.do_actions(actions)

