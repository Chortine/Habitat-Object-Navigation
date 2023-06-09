# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import grpc_clip.clip_text.clip_text_pb2 as clip__text__pb2


class ClipTextServiceStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.function_operate_clip_text = channel.unary_unary(
                '/rpc_package.ClipTextService/function_operate_clip_text',
                request_serializer=clip__text__pb2.text_input.SerializeToString,
                response_deserializer=clip__text__pb2.text_feature.FromString,
                )


class ClipTextServiceServicer(object):
    """Missing associated documentation comment in .proto file"""

    def function_operate_clip_text(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ClipTextServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'function_operate_clip_text': grpc.unary_unary_rpc_method_handler(
                    servicer.function_operate_clip_text,
                    request_deserializer=clip__text__pb2.text_input.FromString,
                    response_serializer=clip__text__pb2.text_feature.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'rpc_package.ClipTextService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ClipTextService(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def function_operate_clip_text(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/rpc_package.ClipTextService/function_operate_clip_text',
            clip__text__pb2.text_input.SerializeToString,
            clip__text__pb2.text_feature.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)