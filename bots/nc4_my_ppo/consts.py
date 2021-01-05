

from enum import Enum
from collections import namedtuple
from sc2.ids.unit_typeid import UnitTypeId


#
#  ProxyEnv와 Actor가 주고 받는 메시지 타입
#

class CommandType(bytes, Enum):
    PING = b'\x00'
    REQ_TASK = b'\x01'
    STATE = b'\x02'
    SCORE = b'\x03'
    ERROR = b'\x04'


class EconomyStrategy(Enum):
    MARINE = UnitTypeId.MARINE
    MARAUDER = UnitTypeId.MARAUDER
    GHOST = UnitTypeId.GHOST
    HELLION = UnitTypeId.HELLION
    SIEGETANK = UnitTypeId.SIEGETANK
    THOR = UnitTypeId.THOR
    MEDIVAC = UnitTypeId.MEDIVAC
    VIKINGFIGHTER = UnitTypeId.VIKINGFIGHTER
    BANSHEE = UnitTypeId.BANSHEE
    RAVEN = UnitTypeId.RAVEN
    BATTLECRUISER = UnitTypeId.BATTLECRUISER
    #MULE = UnitTypeId.MULE # 나중에 추가
    NUKE = UnitTypeId.NUKE # 무적권 마지막에
    #REAPER = UnitTypeId.REAPER

EconomyStrategy.to_index = dict()
EconomyStrategy.to_type_id = dict()

for idx, strategy in enumerate(EconomyStrategy):
    EconomyStrategy.to_index[strategy.value] = idx
    EconomyStrategy.to_type_id[idx] = strategy.value


class ArmyStrategy(Enum):
    DEFENSE = 0
    READY = 1 
    OFFENSE = 2

Sample = namedtuple('Sample', 's, a, r, done, logp, value')

class MessageType(Enum):
    RESULT = 0
    EXCEPTION = 1


# wonseok add #
N_FEATURES = 5 + 13
# wonseok end #
N_ACTIONS = len(EconomyStrategy) * len(ArmyStrategy)
