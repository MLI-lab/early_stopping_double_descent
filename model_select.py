from models import *

class BaseModel():
    @classmethod
    def create(cls, message_type = 'resnet', **kwargs):
        MESSAGE_TYPE_TO_CLASS_MAP = {
            'mcnn' : mcnn.make_cnn,
        }

        if message_type not in MESSAGE_TYPE_TO_CLASS_MAP:
            raise ValueError('Bad message type {}'.format(message_type))

        return MESSAGE_TYPE_TO_CLASS_MAP[message_type](**kwargs)
