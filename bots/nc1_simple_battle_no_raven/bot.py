
__author__ = '박현수 (hspark8312@ncsoft.com), NCSOFT Game AI Lab'

import time
import numpy as np
import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId
from sc2.position import Point2


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

        self.train_raven = False
        self.attacking = False
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first

    async def on_step(self, iteration: int):       
        actions = list()
        #
        # 빌드 오더 생성
        #
        if self.evoked.get('go', False) :
            self.attacking = True
        elif self.units(UnitTypeId.BATTLECRUISER).amount >= 3 :
            if not self.evoked.get('ready', False) :
                self.evoked['ready'] = True
                self.evoked['ready_time'] = self.time
            elif self.time - self.evoked['ready_time'] >= 8.0 : 
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

        if self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
            if self.vespene > 200 and not UnitTypeId.BATTLECRUISER in self.build_order and not UnitTypeId.RAVEN in self.build_order:
                self.build_order.append(UnitTypeId.BATTLECRUISER)
            elif self.vespene <= 200 and not self.build_order:
                self.build_order.insert(0, UnitTypeId.MARINE)


        #
        # 사령부 명령 생성
        #
        if self.build_order and self.can_afford(self.build_order[0]) and self.time - self.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
            # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
            actions.append(self.cc.train(self.build_order[0]))  # 첫 번째 유닛 생산 명령 
            del self.build_order[0]  # 빌드오더에서 첫 번째 유닛 제거
            self.evoked[(self.cc.tag, 'train')] = self.time

        for unit in self.units.not_structure :
            enemy_unit = Point2((32.5, 31.5))
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(Point2((32.5, 31.5))) < unit.distance_to(enemy_unit):
                target = Point2((32.5, 31.5))
            else:
                target = enemy_unit

            if unit.type_id is not UnitTypeId.RAVEN :
                if self.attacking == False :
                    actions.append(unit.attack(Point2((80.0, 31.5))))
                elif self.attacking == True :
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

            ## BATTLECRUISER ##
            if unit.type_id is UnitTypeId.BATTLECRUISER:
                if unit.distance_to(target) < 15 and self.attacking == True:
                    def yamato_target_func(unit):
                        # 야마토 포 상대 지정
                        # 일정 범위 내 적들에 한해 적용
                        yamato_enemy_range = 15
                        yamato_candidate_id = [UnitTypeId.THORAP, UnitTypeId.THOR, UnitTypeId.BATTLECRUISER,
                                                UnitTypeId.SIEGETANKSIEGED,
                                                UnitTypeId.SIEGETANK, UnitTypeId.RAVEN]

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
                            return None

                    # 토르, 밤까마귀, 배틀 같은 성가시거나 피통 많은 애들을 조지는 데 야마토 포 사용
                    # 얘네가 없으면 아껴 놓다가 커맨드에 사용.
                    cruiser_abilities = await self.get_available_abilities(unit)
                    if AbilityId.YAMATO_YAMATOGUN in cruiser_abilities and self.known_enemy_units.exists:
                        yamato_target = yamato_target_func(unit)
                        if yamato_target is None:
                            # 커맨드 unit을 가리키게 변경
                            if not self.known_enemy_units(UnitTypeId.COMMANDCENTER).empty:
                                yamato_target = self.known_enemy_units(UnitTypeId.COMMANDCENTER).first
                        if unit.distance_to(yamato_target) >= 12:
                            actions.append(unit.attack(target))
                        else:
                            actions.append(unit(AbilityId.YAMATO_YAMATOGUN, yamato_target))  # 야마토 박고
                    # 야마토를 쓸 수 없거나 대상이 없다면 주위 위협 제거나 가까운 애들 때리러 간다.
                    else:
                        actions.append(unit.attack(target))

                ## BATTLECRUISER END ##

            # RAVEN
            if unit.type_id is UnitTypeId.RAVEN:
                if self.attacking == True : # 공격중
                    if unit.distance_to(
                            target) < 15 and unit.energy > 75 :  # 적들이 근처에 있고 마나도 있으면
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
                        if self.units.not_structure.exists:  # 전투그룹 중앙 조금 뒤 대기
                            actions.append(unit.move(Point2((self.units.center.x, self.units.center.y - 5))))
                else : # 방어모드
                    actions.append(unit.move(self.cc))

        await self.do_actions(actions)

