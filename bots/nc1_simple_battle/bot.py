
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
        self.tmp_attacking = False

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

    async def on_step(self, iteration: int):

        actions = list()

        if not self.units.not_structure.empty:
            self.my_groups = self.unit_groups()
        #
        # 빌드 오더 생성
        #
        if self.attacking == True and self.units(UnitTypeId.BATTLECRUISER).amount <= 0 :
            self.evoked['go'] = False

        if self.evoked.get('go', False) :
            self.attacking = True
        elif self.units(UnitTypeId.BATTLECRUISER).amount >= 3 :
            if not self.evoked.get('ready', False) :
                self.evoked['ready'] = True
                self.evoked['ready_time'] = self.time
            elif self.time - self.evoked['ready_time'] >= 8.0 : 
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


        if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0: 
            if self.train_raven and self.vespene > 175 and not UnitTypeId.RAVEN in self.build_order and self.units(UnitTypeId.RAVEN).empty: # 상대한테 벤시가 있고 베스핀 175 이상이고 레이븐을 추가한 상태가 아니고 레이븐이 없어야함
                self.build_order = [UnitTypeId.RAVEN]
                #for i in range(0, 10) :
                #    self.build_order.append(UnitTypeId.MARINE)

            elif self.vespene > 200 and not UnitTypeId.BATTLECRUISER in self.build_order and not UnitTypeId.RAVEN in self.build_order:
                self.build_order.append(UnitTypeId.BATTLECRUISER)

            elif self.vespene <= 200 and not self.build_order:
                self.build_order.insert(0, UnitTypeId.MARINE)

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

        closest_dist = 500

        if not self.known_enemy_units.filter(lambda e: e.is_visible).empty:
            for our_unit in self.units:
                temp = self.known_enemy_units.closest_distance_to(our_unit)
                if temp < closest_dist:
                    closest_dist = temp

        # 1등 코드가, 밤까마귀로 하여금 무조건 클라킹한 밴시만 따라다니도록 한다;;
        enemy_banshees = self.known_enemy_units.filter(lambda u: u.type_id is UnitTypeId.BANSHEE)
        #print("enemy_banshees : ", enemy_banshees.amount)

        for unit in self.units.not_structure :

            enemy_units = self.known_enemy_units.filter(lambda u:u.is_visible)
            if enemy_units.exists:
                target = enemy_units.closest_to(self.cc)  # 가장 가까운 적 유닛
            else:
                target = self.enemy_cc

            if not unit.type_id in [UnitTypeId.RAVEN, UnitTypeId.MULE, UnitTypeId.BANSHEE]:
                #if unit.type_id is UnitTypeId.RAVEN:
                #    pass
                #    #print("check")
                is_attacked = False
                visible_enemy = self.known_enemy_units.not_structure.filter(lambda e : e.is_visible)
                for e in visible_enemy :
                    if e.target_in_range(unit) :
                        is_attacked = True
                        break

                if self.attacking == False and (closest_dist > 7.0 and not is_attacked):
                    self.tmp_attacking = False
                    actions.append(unit.attack(self.rally_point))
                else:
                    self.tmp_attacking = True
                    #actions.append(unit.attack(target))

            if unit.type_id is UnitTypeId.MARINE :
                if (self.attacking == True or self.tmp_attacking) :
                    actions.append(unit.attack(target))
                    if unit.distance_to(target) < 15 and self.known_enemy_units.amount >= 3:
                        # 해병과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time

            ## BATTLECRUISER ##
            if unit.type_id is UnitTypeId.BATTLECRUISER:

                if (self.attacking == True or self.tmp_attacking):
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
                    if AbilityId.YAMATO_YAMATOGUN in cruiser_abilities and self.known_enemy_units.exists:
                        yamato_target = yamato_target_func(unit)
                        if yamato_target is None:
                            # 커맨드 unit을 가리키게 변경
                            if not self.known_enemy_units(UnitTypeId.COMMANDCENTER).empty:
                                yamato_target = self.known_enemy_units(UnitTypeId.COMMANDCENTER).first
                            else :
                                actions.append(unit.attack(target))
                                continue
                        if unit.distance_to(yamato_target) >= 12:
                            actions.append(unit.attack(target))
                        else:
                            actions.append(unit(AbilityId.YAMATO_YAMATOGUN, yamato_target))  # 야마토 박고
                    # 야마토를 쓸 수 없거나 대상이 없다면 주위 위협 제거나 가까운 애들 때리러 간다.
                    else:
                        actions.append(unit.attack(target))

                ## BATTLECRUISER END ##

            ## RAVEN ##

            if unit.type_id is UnitTypeId.RAVEN:

                threats = self.select_threat(unit)  # 위협이 있으면 ㅌㅌ
                banshees = self.known_enemy_units(UnitTypeId.BANSHEE).closer_than(unit.sight_range, unit)
                our_auto_turrets = self.units(UnitTypeId.AUTOTURRET)

                if not (self.attacking == True or self.tmp_attacking) \
                        and banshees.exists and unit.energy > 50 and threats.empty:
                    if our_auto_turrets.empty or (
                            not our_auto_turrets.empty and our_auto_turrets.closest_distance_to(unit) < 10):
                        build_loc = banshees.center
                        if await self.can_place(building=AbilityId.BUILDAUTOTURRET_AUTOTURRET,
                                                position=build_loc):
                            actions.append(unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, build_loc))

                elif unit.distance_to(target) < 15 and unit.energy > 75 and \
                        (self.attacking == True or self.tmp_attacking):  # 적들이 근처에 있고 마나도 있으면
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

