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

        # (32.5, 31.5) or (95.5, 31.5)
        if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0:
            self.enemy_cc = Point2(Point2((95.5, 31.5)))  # 적 시작 위치
            self.rally_point = Point2(Point2((47.5, 31.5)))
        else:
            self.enemy_cc = Point2(Point2((32.5, 31.5)))  # 적 시작 위치
            self.rally_point = Point2(Point2((80.5, 31.5)))

    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)

    def select_threat(self, unit):
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

    async def on_step(self, iteration: int):       
        actions = list()

        if not self.units.not_structure.empty:
            my_groups = self.unit_groups()
        #
        # 빌드 오더 생성
        #
        if self.evoked.get('go', False) :
            self.attacking = True
        elif self.units({UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED}).amount >= 6 :
            if not self.evoked.get('ready', False) :
                self.evoked['ready'] = True
                self.evoked['ready_time'] = self.time
            elif self.time - self.evoked['ready_time'] >= 10.0 : 
                if self.units(UnitTypeId.SIEGETANKSIEGED).exists :
                    for unit in self.units(UnitTypeId.SIEGETANKSIEGED) :
                        actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                else :        
                    self.evoked['go'] = True
                    self.attacking = True
        elif self.known_enemy_units.not_structure.exists :
            for unit in self.units :
                if(self.known_enemy_units.not_structure.filter(lambda e : e.is_visible).in_attack_range_of(unit).exists) :
                    self.attacking = True
                    break
                self.attacking = False
        else :
            self.attacking = False
        
        for enemy in self.known_enemy_units :
            if enemy.type_id is UnitTypeId.BANSHEE :
                self.train_raven = True

        if self.time - self.evoked.get('create', 0) > 1.0 and self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0: 
            if self.train_raven and not UnitTypeId.RAVEN in self.build_order and self.units(UnitTypeId.RAVEN).empty: # 상대한테 벤시가 있고 베스핀 175 이상이고 레이븐을 추가한 상태가 아니고 레이븐이 없어야함
                if self.vespene > 175 :
                    self.build_order = [UnitTypeId.RAVEN]
                    self.evoked['create'] = self.time
                elif not self.build_order :
                    self.build_order.insert(0, UnitTypeId.MARINE)
                    self.evoked['create'] = self.time

            elif self.vespene > 85 and not UnitTypeId.SIEGETANK in self.build_order and not UnitTypeId.RAVEN in self.build_order:
                self.build_order.append(UnitTypeId.SIEGETANK)
                self.evoked['create'] = self.time

            elif self.vespene <= 85 and not self.build_order:
                self.build_order.insert(0, UnitTypeId.MARINE)
                self.evoked['create'] = self.time

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 왠지는 모르겠는데 이걸 추가해야 실시간 tracking이 된다..

        #
        # 사령부 명령 생성
        #
        if self.build_order and self.can_afford(self.build_order[0]) and self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
            # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
            actions.append(self.cc.train(self.build_order[0]))  # 첫 번째 유닛 생산 명령 
            del self.build_order[0]  # 빌드오더에서 첫 번째 유닛 제거
            self.evoked[(self.cc.tag, 'train')] = self.time

        closest_dist = 500

        if not self.known_enemy_units.filter(lambda e: e.is_visible).empty:
            for our_unit in self.units:
                temp = self.known_enemy_units.closest_distance_to(our_unit)
                if temp < closest_dist:
                    closest_dist = temp

        # 1등 코드가, 밤까마귀로 하여금 무조건 클라킹한 밴시만 따라다니도록 한다;;
        enemy_banshees = self.known_enemy_units.filter(lambda u: u.type_id is UnitTypeId.BANSHEE)
        print("enemy_banshees : ", enemy_banshees.amount)

        for unit in self.units.not_structure :

            enemy_units = self.known_enemy_units.filter(lambda u: u.is_visible)
            if enemy_units.exists:
                target = enemy_units.closest_to(unit)  # 가장 가까운 적 유닛
            else:
                target = self.enemy_cc

            if not unit.type_id in [UnitTypeId.RAVEN, UnitTypeId.MULE, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED]:

                if self.attacking == False and closest_dist > 7.0:
                    actions.append(unit.attack(self.rally_point))
                else:
                    actions.append(unit.attack(target))

            if unit.type_id is UnitTypeId.MARINE :
                if unit.distance_to(target) < 15 and self.attacking == True:
                    # 해병과 목표의 거리가 15이하일 경우 스팀팩 사용
                    if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                        # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                        if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                            # 1초 이전에 스팀팩을 사용한 적이 없음
                            actions.append(unit(AbilityId.EFFECT_STIM))
                            self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time

            # 시즈탱크
            if unit.type_id is UnitTypeId.SIEGETANK:
                # desired_vector 관련된 정보는 모드에 관계없이 항상 갱신.
                # default : cc.position
                desired_pos = self.cc.position

                if my_groups:
                    # 만약 첫 프레임이거나 이전 프레임에 설정된 그룹 센터와 현재 계산된 그룹 센터가 일정 거리 이상(5)다르다면 이동
                    if self.evoked.get((unit.tag, "desire_add_vector"), None) is None:
                        dist = random.randint(5, 9)
                        dist_x = random.randint(2, dist)
                        dist_y = math.sqrt(dist ** 2 - dist_x ** 2) if random.randint(0, 1) == 0 else -math.sqrt(
                            dist ** 2 - dist_x ** 2)
                        desire_add_vector = Point2((-dist_x, dist_y)) if self.cc.position.x < 50 else Point2((dist_x, dist_y))
                        desired_pos = my_groups[0].center + desire_add_vector
                        desired_pos = Point2((self.clamp(desired_pos.x, 0, self.map_width),
                                                self.clamp(desired_pos.y, 0, self.map_height)))
                        self.evoked[(unit.tag, "group_center")] = my_groups[0].center
                        self.evoked[(unit.tag, "desire_add_vector")] = desire_add_vector
                    else:
                        if my_groups[0].center.distance_to(
                                self.evoked.get((unit.tag, "group_center"), self.cc.position)) > 7:
                            self.evoked[(unit.tag, "group_center")] = my_groups[0].center
                        desired_pos = self.evoked.get((unit.tag, "group_center"), self.cc.position) + self.evoked.get(
                            (unit.tag, "desire_add_vector"), None)

                # 시즈탱크는 공격, 방어 상관없이 기본적으로 항상 정해진 그룹 센터 주변으로 포지셔닝
                # 그룹 센터에서 상대적으로 뒤쪽에 대기한다.
                # 그룹 센터에서 거리는 왼쪽으로 랜덤으로 정해지되, 5-9 정도.
                # 근처 위협이 있다면 무빙샷

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
                    actions.append(unit.attack(target))


            # 시즈모드 시탱
            # 시즈모드일 때에도 공격, 방어모드 상관없이 작동한다.
            if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                if self.known_enemy_units.not_structure.exists:
                    # 타겟팅 정하기
                    target = None
                    # HP 적은 애를 타격하지만, 중장갑 위주
                    targets = self.known_enemy_units.not_structure.filter(lambda u: not u.is_flying and unit.distance_to(u) <= unit.ground_range)
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
                if my_groups:
                    if my_groups[0].center.distance_to(self.evoked.get((unit.tag, "group_center"))) > 7:
                        actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))

            # RAVEN
            if unit.type_id is UnitTypeId.RAVEN:

                if not enemy_banshees.empty:
                    actions.append(unit.move(enemy_banshees.closest_to(unit)))

                if self.attacking == True:  # 공격중
                    if unit.distance_to(target) < 15 and unit.energy > 75:  # 적들이 근처에 있고 마나도 있으면
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
                    else:  # 적들이 없거나 마나가 없으면
                        if enemy_banshees.empty and self.units.not_structure.exists:  # 전투그룹 중앙 조금 뒤 대기
                            if self.cc.position.x < 50:
                                actions.append(
                                    unit.move(Point2((self.units.center.x - 5, self.units.center.y))))
                            else:
                                actions.append(
                                    unit.move(Point2((self.units.center.x + 5, self.units.center.y))))
                elif enemy_banshees.empty:  # 방어모드
                    actions.append(unit.move(self.cc))

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

