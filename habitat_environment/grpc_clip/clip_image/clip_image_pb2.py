# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: clip_image.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='clip_image.proto',
  package='rpc_package',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x10\x63lip_image.proto\x12\x0brpc_package\"8\n\x07raw_rgb\x12\x0e\n\x06my_rgb\x18\x01 \x01(\t\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\"\x1d\n\x0brgb_feature\x12\x0e\n\x06s_list\x18\x01 \x03(\x02\x32X\n\x0b\x43lipService\x12I\n\x15\x66unction_operate_clip\x12\x14.rpc_package.raw_rgb\x1a\x18.rpc_package.rgb_feature\"\x00\x62\x06proto3'
)




_RAW_RGB = _descriptor.Descriptor(
  name='raw_rgb',
  full_name='rpc_package.raw_rgb',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='my_rgb', full_name='rpc_package.raw_rgb.my_rgb', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='width', full_name='rpc_package.raw_rgb.width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='height', full_name='rpc_package.raw_rgb.height', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=33,
  serialized_end=89,
)


_RGB_FEATURE = _descriptor.Descriptor(
  name='rgb_feature',
  full_name='rpc_package.rgb_feature',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s_list', full_name='rpc_package.rgb_feature.s_list', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=91,
  serialized_end=120,
)

DESCRIPTOR.message_types_by_name['raw_rgb'] = _RAW_RGB
DESCRIPTOR.message_types_by_name['rgb_feature'] = _RGB_FEATURE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

raw_rgb = _reflection.GeneratedProtocolMessageType('raw_rgb', (_message.Message,), {
  'DESCRIPTOR' : _RAW_RGB,
  '__module__' : 'clip_image_pb2'
  # @@protoc_insertion_point(class_scope:rpc_package.raw_rgb)
  })
_sym_db.RegisterMessage(raw_rgb)

rgb_feature = _reflection.GeneratedProtocolMessageType('rgb_feature', (_message.Message,), {
  'DESCRIPTOR' : _RGB_FEATURE,
  '__module__' : 'clip_image_pb2'
  # @@protoc_insertion_point(class_scope:rpc_package.rgb_feature)
  })
_sym_db.RegisterMessage(rgb_feature)



_CLIPSERVICE = _descriptor.ServiceDescriptor(
  name='ClipService',
  full_name='rpc_package.ClipService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=122,
  serialized_end=210,
  methods=[
  _descriptor.MethodDescriptor(
    name='function_operate_clip',
    full_name='rpc_package.ClipService.function_operate_clip',
    index=0,
    containing_service=None,
    input_type=_RAW_RGB,
    output_type=_RGB_FEATURE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_CLIPSERVICE)

DESCRIPTOR.services_by_name['ClipService'] = _CLIPSERVICE

# @@protoc_insertion_point(module_scope)
