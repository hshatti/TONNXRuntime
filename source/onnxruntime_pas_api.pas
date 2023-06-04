{ Onnxruntime for FreePascal/Delphi

  Copyright (c) 2022 Haitham Shatti <haitham.shatti at gmail dot com> <https://github.com/hshatti>

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
}

unit onnxruntime_pas_api;
interface
  {$IFDEF FPC}
    {$mode delphi}
    {$ModeSwitch advancedrecords}
    {$ModeSwitch typehelpers}
    {.$linklib onnxruntime.dll}
    {$MACRO ON}
    {$PACKRECORDS C}
    {$PackEnum 4}
  {$else}
    {$Z4}
    type size_t=UInt64; // delphi?

  {$ENDIF}
  {$define ORT_API_CALL:=stdcall}

  {$ifdef MSWINDOWS}
  const dllname='onnxruntime.dll';
  {$endif}

  {$ifdef darwin}
  const dllname='onnxruntime.dylib';
  {$endif}
  {$ifdef linux}
  const dllname='onnxruntime.so';
  {$endif}
  {$ifdef ONNX_NEW_VERSION}
  const ORT_API_VERSION = 13;
  {$else}
  const ORT_API_VERSION = 10;
  {$endif}

{$ifndef SIZE_MAX}
  {$ifdef WIN64}
    const SIZE_MAX = $ffffffffffffffff;
  {$else}
    const SIZE_MAX : size_t = $ffffffff;
  {$endif}
{$endif}

(*
 * The following defines SessionOptions Config Keys and format of the Config Values.
 *
 * The Naming Convention for a SessionOptions Config Key,
 * "[Area][.[SubArea1].[SubArea2]...].[Keyname]"
 * Such as "ep.cuda.use_arena"
 * The Config Key cannot be empty
 * The maximum length of the Config Key is 128
 *
 * The string format of a SessionOptions Config Value is defined individually for each Config.
 * The maximum length of the Config Value is 1024
 *)


      // Key for enabling shrinkages of user listed device memory arenas.
      // Expects a list of semi-colon separated key value pairs separated by colon in the following format:
      // "device_0:device_id_0;device_1:device_id_1"
      // No white-spaces allowed in the provided list string.
      // Currently, the only supported devices are : "cpu", "gpu" (case sensitive).
      // If "cpu" is included in the list, DisableCpuMemArena() API must not be called (i.e.) arena for cpu should be enabled.// Example usage: "cpu:0;gpu:0" (or) "gpu:0"
      // By default, the value for this key is empty (i.e.) no memory arenas are shrunk
      kOrtRunOptionsConfigEnableMemoryArenaShrinkage = 'memory.enable_memory_arena_shrinkage';



      // Key for disable PrePacking,
      // If the config value is set to "1" then the prepacking is disabled, otherwise prepacking is enabled (default value)
      kOrtSessionOptionsConfigDisablePrepacking = 'session.disable_prepacking';

      // A value of "1" means allocators registered in the env will be used. "0" means the allocators created in the session
      // will be used. Use this to override the usage of env allocators on a per session level.
      kOrtSessionOptionsConfigUseEnvAllocators = 'session.use_env_allocators';

      // Set to 'ORT' (case sensitive) to load an ORT format model.
      // If unset, model type will default to ONNX unless inferred from filename ('.ort' == ORT format) or bytes to be ORT
      kOrtSessionOptionsConfigLoadModelFormat = 'session.load_model_format';

      // Set to 'ORT' (case sensitive) to save optimized model in ORT format when SessionOptions.optimized_model_path is set.
      // If unset, format will default to ONNX unless optimized_model_filepath ends in '.ort'.
      kOrtSessionOptionsConfigSaveModelFormat = 'session.save_model_format';

      // If a value is "1", flush-to-zero and denormal-as-zero are applied. The default is "0".
      // When multiple sessions are created, a main thread doesn't override changes from succeeding session options,
      // but threads in session thread pools follow option changes.
      // When ORT runs with OpenMP, the same rule is applied, i.e. the first session option to flush-to-zero and
      // denormal-as-zero is only applied to global OpenMP thread pool, which doesn't support per-session thread pool.
      // Note that an alternative way not using this option at runtime is to train and export a model without denormals
      // and that's recommended because turning this option on may hurt model accuracy.
      kOrtSessionOptionsConfigSetDenormalAsZero = 'session.set_denormal_as_zero';

      // It controls to run quantization model in QDQ (QuantizelinearDeQuantizelinear) format or not.
      // "0": enable. ORT does fusion logic for QDQ format.
      // "1": disable. ORT doesn't do fusion logic for QDQ format.
      // Its default value is "0"
      kOrtSessionOptionsDisableQuantQDQ = 'session.disable_quant_qdq';

      // If set to "1", enables the removal of QuantizeLinear/DequantizeLinear node pairs once all QDQ handling has been
      // completed. e.g. If after all QDQ handling has completed and we have -> FloatOp -> Q -> DQ -> FloatOp -> the
      // Q -> DQ could potentially be removed. This will provide a performance benefit by avoiding going from float to
      // 8-bit and back to float, but could impact accuracy. The impact on accuracy will be model specific and depend on
      // other factors like whether the model was created using Quantization Aware Training or Post Training Quantization.
      // As such, it's best to test to determine if enabling this works well for your scenario.
      // The default value is "0"
      // Available since version 1.11.
      kOrtSessionOptionsEnableQuantQDQCleanup = 'session.enable_quant_qdq_cleanup';

      // Enable or disable gelu approximation in graph optimization. "0": disable; "1": enable. The default is "0".
      // GeluApproximation has side effects which may change the inference results. It is disabled by default due to this.
      kOrtSessionOptionsEnableGeluApproximation = 'optimization.enable_gelu_approximation';

      // Enable or disable using device allocator for allocating initialized tensor memory. "1": enable; "0": disable. The default is "0".
      // Using device allocators means the memory allocation is made using malloc/new.
      kOrtSessionOptionsUseDeviceAllocatorForInitializers = 'session.use_device_allocator_for_initializers';

      // Configure whether to allow the inter_op/intra_op threads spinning a number of times before blocking
      // "0": thread will block if found no job to run
      // "1": default, thread will spin a number of times before blocking
      kOrtSessionOptionsConfigAllowInterOpSpinning = 'session.inter_op.allow_spinning';
      kOrtSessionOptionsConfigAllowIntraOpSpinning = 'session.intra_op.allow_spinning';

      // Key for using model bytes directly for ORT format
      // If a session is created using an input byte array contains the ORT format model data,
      // By default we will copy the model bytes at the time of session creation to ensure the model bytes
      // buffer is valid.
      // Setting this option to "1" will disable copy the model bytes, and use the model bytes directly. The caller
      // has to guarantee that the model bytes are valid until the ORT session using the model bytes is destroyed.
      kOrtSessionOptionsConfigUseORTModelBytesDirectly = 'session.use_ort_model_bytes_directly';

      /// <summary>
      /// Key for using the ORT format model flatbuffer bytes directly for initializers.
      /// This avoids copying the bytes and reduces peak memory usage during model loading and initialization.
      /// Requires `session.use_ort_model_bytes_directly` to be true.
      /// If set, the flatbuffer bytes provided when creating the InferenceSession MUST remain valid for the entire
      /// duration of the InferenceSession.
      /// </summary>
      kOrtSessionOptionsConfigUseORTModelBytesForInitializers =
          'session.use_ort_model_bytes_for_initializers';

      // This should only be specified when exporting an ORT format model for use on a different platform.
      // If the ORT format model will be used on ARM platforms set to "1". For other platforms set to "0"
      // Available since version 1.11.
      kOrtSessionOptionsQDQIsInt8Allowed = 'session.qdqisint8allowed';

      // x64 SSE4.1/AVX2/AVX512(with no VNNI) has overflow problem with quantizied matrix multiplication with U8S8.
      // To avoid this we need to use slower U8U8 matrix multiplication instead. This option, if
      // turned on, use slower U8U8 matrix multiplications. Only effective with AVX2 or AVX512
      // platforms.
      kOrtSessionOptionsAvx2PrecisionMode = 'session.x64quantprecision';

      // Specifies how minimal build graph optimizations are handled in a full build.
      // These optimizations are at the extended level or higher.
      // Possible values and their effects are:
      // "save": Save runtime optimizations when saving an ORT format model.
      // "apply": Only apply optimizations available in a minimal build.
      // ""/<unspecified>: Apply optimizations available in a full build.
      // Available since version 1.11.
      kOrtSessionOptionsConfigMinimalBuildOptimizations =
          'optimization.minimal_build_optimizations';

      // Note: The options specific to an EP should be specified prior to appending that EP to the session options object in
      // order for them to take effect.

      // Specifies a list of stop op types. Nodes of a type in the stop op types and nodes downstream from them will not be
      // run by the NNAPI EP.
      // The value should be a ","-delimited list of op types. For example, "Add,Sub".
      // If not specified, the default set of stop ops is used. To specify an empty stop ops types list and disable stop op
      // exclusion, set the value to "".
      kOrtSessionOptionsConfigNnapiEpPartitioningStopOps = 'ep.nnapi.partitioning_stop_ops';

      // Enabling dynamic block-sizing for multithreading.
      // With a positive value, thread pool will split a task of N iterations to blocks of size starting from:
      // N / (num_of_threads * dynamic_block_base)
      // As execution progresses, the size will decrease according to the diminishing residual of N,
      // meaning the task will be distributed in smaller granularity for better parallelism.
      // For some models, it helps to reduce the variance of E2E inference latency and boost performance.
      // The feature will not function by default, specify any positive integer, e.g. "4", to enable it.
      // Available since version 1.11.
      kOrtSessionOptionsConfigDynamicBlockBase = 'session.dynamic_block_base';

      // This option allows to decrease CPU usage between infrequent
      // requests and forces any TP threads spinning stop immediately when the last of
      // concurrent Run() call returns.
      // Spinning is restarted on the next Run() call.
      // Applies only to internal thread-pools
      kOrtSessionOptionsConfigForceSpinningStop = 'session.force_spinning_stop';

      // "1": all inconsistencies encountered during shape and type inference
      // will result in failures.
      // "0": in some cases warnings will be logged but processing will continue. The default.
      // May be useful to expose bugs in models.
      kOrtSessionOptionsConfigStrictShapeTypeInference = 'session.strict_shape_type_inference';


  Type

  PPPOrtChar = ^PPOrtChar;
  PPOrtChar = ^POrtChar;
  POrtChar = PAnsiChar;
  PORTCHAR_T  = ^ORTCHAR_T;
{$ifdef MSWINDOWS}
    wchar_t = WideChar;
    ORTCHAR_T = wchar_t;
    OrtChar=ansichar;
{$else}
    ORTCHAR_T = ansichar;
    OrtChar   = ansichar;
{$endif}
  //Pchar  = ^char;
  Pint32_t  = ^int32_t;
  Pint64_t  = ^int64_t;
  Plongint  = ^longint;
  PONNXTensorElementDataType  = ^ONNXTensorElementDataType;
  PONNXType  = ^ONNXType;
  PPOrtAllocator = ^POrtAllocator;
  POrtAllocator  = ^OrtAllocator;
  POrtAllocatorType  = ^OrtAllocatorType;
  POrtApi  = ^OrtApi;
  POrtApiBase  = ^OrtApiBase;
  PPOrtArenaCfg = ^POrtArenaCfg;
  POrtArenaCfg  = ^OrtArenaCfg;
  PPOrtCANNProviderOptions = ^POrtCANNProviderOptions;
  POrtCANNProviderOptions  = ^OrtCANNProviderOptions;
  PPOrtCUDAProviderOptions = ^POrtCUDAProviderOptions;
  POrtCUDAProviderOptions  = ^OrtCUDAProviderOptions;
  PPOrtCUDAProviderOptionsV2 = ^POrtCUDAProviderOptionsV2;
  POrtCUDAProviderOptionsV2  = ^OrtCUDAProviderOptionsV2;
  POrtCustomOp  = ^OrtCustomOp;
  PPOrtCustomOpDomain = ^POrtCustomOpDomain;
  POrtCustomOpDomain  = ^OrtCustomOpDomain;
  PPOrtEnv = ^POrtEnv;
  POrtEnv  = ^OrtEnv;
  PPOrtIoBinding = ^POrtIoBinding;
  POrtIoBinding  = ^OrtIoBinding;
  POrtKernelContext  = ^OrtKernelContext;
  PPOrtKernelInfo = ^POrtKernelInfo;
  POrtKernelInfo  = ^OrtKernelInfo;
  PPOrtMapTypeInfo = ^POrtMapTypeInfo;
  POrtMapTypeInfo  = ^OrtMapTypeInfo;
  PPOrtMemoryInfo = ^POrtMemoryInfo;
  POrtMemoryInfo  = ^OrtMemoryInfo;
  POrtMemType  = ^OrtMemType;
  POrtMemoryInfoDeviceType = ^OrtMemoryInfoDeviceType;
  POrtMIGraphXProviderOptions  = ^OrtMIGraphXProviderOptions;
  PPOrtModelMetadata = ^POrtModelMetadata;
  POrtModelMetadata  = ^OrtModelMetadata;
  PPOrtOp = ^POrtOp;
  POrtOp  = ^OrtOp;
  PPOrtOpAttr =^POrtOpAttr;
  POrtOpAttr  = ^OrtOpAttr;
  POrtOpenVINOProviderOptions  = ^OrtOpenVINOProviderOptions;
  PPOrtPrepackedWeightsContainer = ^POrtPrepackedWeightsContainer;
  POrtPrepackedWeightsContainer  = ^OrtPrepackedWeightsContainer;
  POrtROCMProviderOptions  = ^OrtROCMProviderOptions;
  PPOrtRunOptions = ^POrtRunOptions;
  POrtRunOptions  = ^OrtRunOptions;
  PPOrtSequenceTypeInfo = ^POrtSequenceTypeInfo;
  POrtSequenceTypeInfo  = ^OrtSequenceTypeInfo;
  PPOrtSession = ^ POrtSession;
  POrtSession  = ^OrtSession;
  PPOrtSessionOptions = ^POrtSessionOptions;
  POrtSessionOptions  = ^OrtSessionOptions;
  POrtSparseFormat  = ^OrtSparseFormat;
  POrtStatus  = ^OrtStatus;
  PPOrtTensorRTProviderOptionsV2 = ^POrtTensorRTProviderOptionsV2;
  POrtTensorRTProviderOptions  = ^OrtTensorRTProviderOptions;
  POrtTensorRTProviderOptionsV2  = ^OrtTensorRTProviderOptionsV2;
  PPOrtTensorTypeAndShapeInfo = ^POrtTensorTypeAndShapeInfo;
  POrtTensorTypeAndShapeInfo  = ^OrtTensorTypeAndShapeInfo;
  PPOrtThreadingOptions = ^POrtThreadingOptions;
  POrtThreadingOptions  = ^OrtThreadingOptions;
  POrtTrainingApi  = ^OrtTrainingApi;
  PPOrtTypeInfo = ^POrtTypeInfo;
  POrtTypeInfo  = ^OrtTypeInfo;
  PPPOrtValue = ^PPOrtValue;
  PPOrtValue = ^POrtValue;
  POrtValue  = ^OrtValue;
  Psingle  = ^single;
  PPSize_t = ^PSize_t;
  Psize_t  = ^size_t;
  Puint64_t  = ^uint64_t;

  int8_t   = shortint;
  int16_t  = SmallInt;
  int32_t  = longint ;
  int64_t  = int64   ;

  uint8_t  = byte    ;
  uint16_t = word    ;
  uint32_t = longword;
  uint64_t = UInt64   ;


  ONNXTensorElementDataType = (ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
    );

  ONNXType = (ONNX_TYPE_UNKNOWN,ONNX_TYPE_TENSOR,ONNX_TYPE_SEQUENCE,
    ONNX_TYPE_MAP,ONNX_TYPE_OPAQUE,ONNX_TYPE_SPARSETENSOR,
    ONNX_TYPE_OPTIONAL);
  OrtSparseFormat = (ORT_SPARSE_UNDEFINED = 0,ORT_SPARSE_COO = $1,
    ORT_SPARSE_CSRC = $2,ORT_SPARSE_BLOCK_SPARSE = $4
    );
  OrtSparseIndicesFormat = (ORT_SPARSE_COO_INDICES,ORT_SPARSE_CSR_INNER_INDICES,
    ORT_SPARSE_CSR_OUTER_INDICES,ORT_SPARSE_BLOCK_SPARSE_INDICES
    );


  OrtLoggingLevel = (ORT_LOGGING_LEVEL_VERBOSE,ORT_LOGGING_LEVEL_INFO,
    ORT_LOGGING_LEVEL_WARNING,ORT_LOGGING_LEVEL_ERROR,
    ORT_LOGGING_LEVEL_FATAL);

  OrtErrorCode = (ORT_OK,ORT_FAIL,ORT_INVALID_ARGUMENT,ORT_NO_SUCHFILE,
    ORT_NO_MODEL,ORT_ENGINE_ERROR,ORT_RUNTIME_EXCEPTION,
    ORT_INVALID_PROTOBUF,ORT_MODEL_LOADED,
    ORT_NOT_IMPLEMENTED,ORT_INVALID_GRAPH,
    ORT_EP_FAIL);

  OrtOpAttrType = (ORT_OP_ATTR_UNDEFINED = 0,ORT_OP_ATTR_INT,
    ORT_OP_ATTR_INTS,ORT_OP_ATTR_FLOAT,ORT_OP_ATTR_FLOATS,
    ORT_OP_ATTR_STRING,ORT_OP_ATTR_STRINGS
    );
  OrtEnv = record
    end;

  OrtStatus = record
    end;

  OrtMemoryInfo = record
    end;

  OrtIoBinding = record
    end;


  OrtSession = record
    end;


  OrtValue = record
    end;


  OrtRunOptions = record
    end;


  OrtTypeInfo = record
    end;


  OrtTensorTypeAndShapeInfo = record
    end;


  OrtSessionOptions = record
    end;


  OrtCustomOpDomain = record
    end;


  OrtMapTypeInfo = record
    end;


  OrtSequenceTypeInfo = record
    end;


  OrtModelMetadata = record
    end;


  OrtThreadPoolParams = record
    end;


  OrtThreadingOptions = record
    end;


  OrtArenaCfg = record
    end;


  OrtPrepackedWeightsContainer = record
    end;


  OrtTensorRTProviderOptionsV2 = record
    end;


  OrtCUDAProviderOptionsV2 = record
    end;


  OrtCANNProviderOptions = record
    end;


  OrtOp = record
    end;


  OrtOpAttr = record
    end;

  OrtAllocator = record
      version : uint32_t;
      Alloc : function (this_:POrtAllocator; size:size_t):pointer;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
      Free : procedure (this_:POrtAllocator; p:pointer);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
      Info : function (this_:POrtAllocator):POrtMemoryInfo;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
  end;

 OrtLoggingFunction = procedure (param:pointer; severity:OrtLoggingLevel; category:POrtChar; logid:POrtChar; code_location:POrtChar;
               message:POrtChar);cdecl;

 GraphOptimizationLevel = (ORT_DISABLE_ALL = 0,ORT_ENABLE_BASIC = 1,
   ORT_ENABLE_EXTENDED = 2,ORT_ENABLE_ALL = 99
   );

 ExecutionMode = (ORT_SEQUENTIAL = 0,ORT_PARALLEL = 1
   );

 OrtLanguageProjection = (ORT_PROJECTION_C = 0,ORT_PROJECTION_CPLUSPLUS = 1,
   ORT_PROJECTION_CSHARP = 2,ORT_PROJECTION_PYTHON = 3,
   ORT_PROJECTION_JAVA = 4,ORT_PROJECTION_WINML = 5,
   ORT_PROJECTION_NODEJS = 6);

 OrtKernelInfo = record
   end;

 OrtKernelContext = record
   end;

 //OrtCustomOp = record
 //  end;
 //

 OrtAllocatorType = (OrtInvalidAllocator = -1,OrtDeviceAllocator = 0,
   OrtArenaAllocator = 1);

 OrtMemType = (OrtMemTypeCPUInput = -2,OrtMemTypeCPUOutput = -1,
   OrtMemTypeCPU = OrtMemTypeCPUOutput,OrtMemTypeDefault = 0
   );

 OrtMemoryInfoDeviceType = (
  OrtMemoryInfoDeviceType_CPU = 0,
  OrtMemoryInfoDeviceType_GPU = 1,
  OrtMemoryInfoDeviceType_FPGA = 2
 );

 OrtCudnnConvAlgoSearch = (OrtCudnnConvAlgoSearchExhaustive,OrtCudnnConvAlgoSearchHeuristic,
   OrtCudnnConvAlgoSearchDefault);


 { OrtCUDAProviderOptions }

 OrtCUDAProviderOptions = record
     device_id : longint;
     cudnn_conv_algo_search : OrtCudnnConvAlgoSearch;
     gpu_mem_limit : size_t;
     arena_extend_strategy : longint;
     do_copy_in_default_stream : longint;
     has_user_compute_stream : longint;
     user_compute_stream : pointer;
     default_memory_arena_cfg : ^OrtArenaCfg;
     class operator Initialize({$ifdef fpc}var{$else}out{$endif}dest:OrtCUDAProviderOptions);
   end;

 { OrtROCMProviderOptions }

 OrtROCMProviderOptions = record
     device_id : longint;
     miopen_conv_exhaustive_search : longint;
     gpu_mem_limit : size_t;
     arena_extend_strategy : longint;
     do_copy_in_default_stream : longint;
     has_user_compute_stream : longint;
     user_compute_stream : pointer;
     default_memory_arena_cfg : ^OrtArenaCfg;
     class operator Initialize({$ifdef fpc}var{$else}out{$endif}dest:OrtROCMProviderOptions);
   end;

 OrtTensorRTProviderOptions = record
     device_id : longint;
     has_user_compute_stream : longint;
     user_compute_stream : pointer;
     trt_max_partition_iterations : longint;
     trt_min_subgraph_size : longint;
     trt_max_workspace_size : size_t;
     trt_fp16_enable : longint;
     trt_int8_enable : longint;
     trt_int8_calibration_table_name : ^char;
     trt_int8_use_native_calibration_table : longint;
     trt_dla_enable : longint;
     trt_dla_core : longint;
     trt_dump_subgraphs : longint;
     trt_engine_cache_enable : longint;
     trt_engine_cache_path : ^char;
     trt_engine_decryption_enable : longint;
     trt_engine_decryption_lib_path : ^char;
     trt_force_sequential_engine_build : longint;
   end;

 OrtMIGraphXProviderOptions = record
     device_id : longint;
     migraphx_fp16_enable : longint;
     migraphx_int8_enable : longint;
   end;

 OrtOpenVINOProviderOptions = record
     device_type : ^char;
     enable_vpu_fast_compile : byte;
     device_id : ^char;
     num_of_threads : size_t;
     use_compiled_network : byte;
     blob_dump_path : ^char;
     context : pointer;
     enable_opencl_throttling : byte;
     enable_dynamic_shapes : byte;
   end;

 OrtTrainingApi = record
   end;

 OrtApiBase = record
     GetApi : function (version:uint32_t):POrtApi;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetVersionString : function :POrtChar;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   end;


 OrtThreadWorkerFn = procedure (ort_worker_fn_param:pointer);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

 OrtCustomHandleType = record
     __place_holder : char;
   end;
 OrtCustomThreadHandle = ^OrtCustomHandleType;

 OrtCustomCreateThreadFn = function (ort_custom_thread_creation_options:pointer; ort_thread_worker_fn:OrtThreadWorkerFn; ort_worker_fn_param:pointer):OrtCustomThreadHandle;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

 OrtCustomJoinThreadFn = procedure (ort_custom_thread_handle:OrtCustomThreadHandle);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

 OrtApi = record
   CreateStatus : function (code:OrtErrorCode; const msg:POrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetErrorCode : function ( const status:POrtStatus):OrtErrorCode;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetErrorMessage : function (const status:POrtStatus):POrtChar;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateEnv : function (log_severity_level:OrtLoggingLevel;const  logid:POrtChar; _out:PPOrtEnv):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateEnvWithCustomLogger : function (logging_function:OrtLoggingFunction; logger_param:pointer; log_severity_level:OrtLoggingLevel;const  logid:POrtChar; _out:PPOrtEnv):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   EnableTelemetryEvents : function (const env:POrtEnv):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   DisableTelemetryEvents : function (const env:POrtEnv):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateSession : function (const env:POrtEnv;const  model_path:PORTCHAR_T;const  options:POrtSessionOptions;const  _out:PPOrtSession):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateSessionFromArray : function (const env:POrtEnv;const  model_data:pointer;const  model_data_length:size_t;const  options:POrtSessionOptions; _out:PPOrtSession):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   Run : function (session:POrtSession;const  run_options:POrtRunOptions;const  input_names:PPOrtChar;const inputs:PPOrtValue; input_len:size_t;const output_names:PPOrtChar; output_names_len:size_t; outputs:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateSessionOptions : function (options:PPOrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetOptimizedModelFilePath : function (options:POrtSessionOptions;const optimized_model_filepath:PORTCHAR_T):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CloneSessionOptions : function (const in_options:POrtSessionOptions; out_options:PPOrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetSessionExecutionMode : function (options:POrtSessionOptions; execution_mode:ExecutionMode):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   EnableProfiling : function (options:POrtSessionOptions;const profile_file_prefix:PORTCHAR_T):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   DisableProfiling : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   EnableMemPattern : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   DisableMemPattern : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   EnableCpuMemArena : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   DisableCpuMemArena : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetSessionLogId : function (options:POrtSessionOptions;const logid:POrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetSessionLogVerbosityLevel : function (options:POrtSessionOptions; session_log_verbosity_level:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetSessionLogSeverityLevel : function (options:POrtSessionOptions; session_log_severity_level:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetSessionGraphOptimizationLevel : function (options:POrtSessionOptions; graph_optimization_level:GraphOptimizationLevel):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetIntraOpNumThreads : function (options:POrtSessionOptions; intra_op_num_threads:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetInterOpNumThreads : function (options:POrtSessionOptions; inter_op_num_threads:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateCustomOpDomain : function (const domain:POrtChar; _out:PPOrtCustomOpDomain):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CustomOpDomain_Add : function (custom_op_domain:POrtCustomOpDomain;const op:POrtCustomOp):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AddCustomOpDomain : function (options:POrtSessionOptions; custom_op_domain:POrtCustomOpDomain):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RegisterCustomOpsLibrary : function (options:POrtSessionOptions;const library_path:POrtChar; library_handle:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetInputCount : function (const session:POrtSession; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetOutputCount : function (const session:POrtSession; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetOverridableInitializerCount : function (const session:POrtSession; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetInputTypeInfo : function (const session:POrtSession; index:size_t; type_info:PPOrtTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetOutputTypeInfo : function (const session:POrtSession; index:size_t; type_info:PPOrtTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetOverridableInitializerTypeInfo : function (const session:POrtSession; index:size_t; type_info:PPOrtTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetInputName : function (const session:POrtSession; index:size_t; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetOutputName : function (const session:POrtSession; index:size_t; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetOverridableInitializerName : function (const session:POrtSession; index:size_t; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateRunOptions : function (_out:PPOrtRunOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsSetRunLogVerbosityLevel : function (options:POrtRunOptions; log_verbosity_level:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsSetRunLogSeverityLevel : function (options:POrtRunOptions; log_severity_level:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsSetRunTag : function (options:POrtRunOptions;const run_tag:POrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsGetRunLogVerbosityLevel : function (const options:POrtRunOptions; log_verbosity_level:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsGetRunLogSeverityLevel : function (const options:POrtRunOptions; log_severity_level:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsGetRunTag : function (const options:POrtRunOptions;const  run_tag:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsSetTerminate : function (options:POrtRunOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunOptionsUnsetTerminate : function (options:POrtRunOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateTensorAsOrtValue : function (allocator:POrtAllocator;const shape:Pint64_t; shape_len:size_t; _type:ONNXTensorElementDataType; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateTensorWithDataAsOrtValue : function (const info:POrtMemoryInfo; p_data:pointer; p_data_len:size_t;const shape:Pint64_t; shape_len:size_t; _type:ONNXTensorElementDataType; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   IsTensor : function (const value:POrtValue; _out:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetTensorMutableData : function (value:POrtValue; _out:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   FillStringTensor : function (value:POrtValue;const s:PPOrtChar; s_len:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetStringTensorDataLength : function (const value:POrtValue; len:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetStringTensorContent : function (const value:POrtValue; s:pointer; s_len:size_t; offsets:Psize_t; offsets_len:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CastTypeInfoToTensorInfo : function (const type_info:POrtTypeInfo;const  _out:PPOrtTensorTypeAndShapeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetOnnxTypeFromTypeInfo : function (const type_info:POrtTypeInfo; _out:PONNXType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateTensorTypeAndShapeInfo : function (_out:PPOrtTensorTypeAndShapeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetTensorElementType : function (info:POrtTensorTypeAndShapeInfo; _type:ONNXTensorElementDataType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetDimensions : function (info:POrtTensorTypeAndShapeInfo;const  dim_values:Pint64_t; dim_count:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetTensorElementType : function (const info:POrtTensorTypeAndShapeInfo; _out:PONNXTensorElementDataType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetDimensionsCount : function (const info:POrtTensorTypeAndShapeInfo; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetDimensions : function (const info:POrtTensorTypeAndShapeInfo; dim_values:Pint64_t; dim_values_length:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetSymbolicDimensions : function (const info:POrtTensorTypeAndShapeInfo; const dim_params:PPOrtChar; dim_params_length:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetTensorShapeElementCount : function (const info:POrtTensorTypeAndShapeInfo; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetTensorTypeAndShape : function (const value:POrtValue; _out:PPOrtTensorTypeAndShapeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetTypeInfo : function (const value:POrtValue; _out:PPOrtTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetValueType : function (const value:POrtValue; _out:PONNXType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateMemoryInfo : function (const name:POrtChar; _type:OrtAllocatorType; id:longint; mem_type:OrtMemType; _out:PPOrtMemoryInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateCpuMemoryInfo : function (_type:OrtAllocatorType; mem_type:OrtMemType; _out:PPOrtMemoryInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CompareMemoryInfo : function (const info1:POrtMemoryInfo;const info2:POrtMemoryInfo; _out:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   MemoryInfoGetName : function (const ptr:POrtMemoryInfo;const _out:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   MemoryInfoGetId : function (const ptr:POrtMemoryInfo; _out:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   MemoryInfoGetMemType : function (const ptr:POrtMemoryInfo; _out:POrtMemType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   MemoryInfoGetType : function (const ptr:POrtMemoryInfo; _out:POrtAllocatorType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AllocatorAlloc : function (ort_allocator:POrtAllocator; size:size_t; _out:PPointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AllocatorFree : function (ort_allocator:POrtAllocator; p:pointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AllocatorGetInfo : function (const ort_allocator:POrtAllocator;const _out:PPOrtMemoryInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetAllocatorWithDefaultOptions : function (_out:PPOrtAllocator):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AddFreeDimensionOverride : function (options:POrtSessionOptions;const  dim_denotation:POrtChar; dim_value:int64_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetValue : function (const value:POrtValue; index:longint; allocator:POrtAllocator; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetValueCount : function (const value:POrtValue; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateValue : function (const _in:PPOrtValue; num_values:size_t; value_type:ONNXType; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateOpaqueValue : function (const domain_name:POrtChar;const type_name:POrtChar;const data_container:pointer; data_container_size:size_t; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetOpaqueValue : function (const domain_name:POrtChar;const type_name:POrtChar;const _in:POrtValue; data_container:pointer; data_container_size:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   KernelInfoGetAttribute_float : function (const info:POrtKernelInfo;const name:POrtChar; _out:Psingle):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelInfoGetAttribute_int64 : function (const info:POrtKernelInfo;const name:POrtChar; _out:Pint64_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelInfoGetAttribute_string : function (const info:POrtKernelInfo;const name:POrtChar; _out:POrtChar; size:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   KernelContext_GetInputCount : function (const context:POrtKernelContext; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelContext_GetOutputCount : function (const context:POrtKernelContext; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelContext_GetInput : function (const context:POrtKernelContext; index:size_t;const _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelContext_GetOutput : function (context:POrtKernelContext; index:size_t;const dim_values:Pint64_t; dim_count:size_t; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseEnv : procedure (input:POrtEnv);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseStatus : procedure (input:POrtStatus);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseMemoryInfo : procedure (input:POrtMemoryInfo);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseSession : procedure (input:POrtSession);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseValue : procedure (input:POrtValue);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseRunOptions : procedure (input:POrtRunOptions);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseTypeInfo : procedure (input:POrtTypeInfo);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseTensorTypeAndShapeInfo : procedure (input:POrtTensorTypeAndShapeInfo);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseSessionOptions : procedure (input:POrtSessionOptions);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseCustomOpDomain : procedure (input:POrtCustomOpDomain);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetDenotationFromTypeInfo : function (const type_info:POrtTypeInfo;const denotation:PPOrtChar; len:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CastTypeInfoToMapTypeInfo : function (const type_info:POrtTypeInfo;const _out:PPOrtMapTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CastTypeInfoToSequenceTypeInfo : function (const type_info:POrtTypeInfo;const _out:PPOrtSequenceTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetMapKeyType : function (const map_type_info:POrtMapTypeInfo; _out:PONNXTensorElementDataType):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetMapValueType : function (const map_type_info:POrtMapTypeInfo; type_info:PPOrtTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetSequenceElementType : function (const sequence_type_info:POrtSequenceTypeInfo; type_info:PPOrtTypeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ReleaseMapTypeInfo : procedure (input:POrtMapTypeInfo);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseSequenceTypeInfo : procedure (input:POrtSequenceTypeInfo);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionEndProfiling : function (session:POrtSession; allocator:POrtAllocator; _out:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionGetModelMetadata : function (const session:POrtSession; _out:PPOrtModelMetadata):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataGetProducerName : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataGetGraphName : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataGetDomain : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataGetDescription : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataLookupCustomMetadataMap : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator;const  key:POrtChar; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataGetVersion : function (const model_metadata:POrtModelMetadata; value:Pint64_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ReleaseModelMetadata : procedure (input:POrtModelMetadata);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateEnvWithGlobalThreadPools : function (log_severity_level:OrtLoggingLevel;const logid:POrtChar;const tp_options:POrtThreadingOptions; _out:PPOrtEnv):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   DisablePerSessionThreads : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateThreadingOptions : function (_out:PPOrtThreadingOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseThreadingOptions : procedure (input:POrtThreadingOptions);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ModelMetadataGetCustomMetadataMapKeys : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator; keys:PPPOrtChar; num_keys:Pint64_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AddFreeDimensionOverrideByName : function (options:POrtSessionOptions;const  dim_name:POrtChar; dim_value:int64_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetAvailableProviders : function (out_ptr:PPPOrtChar; provider_length:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseAvailableProviders : function (ptr:PPOrtChar; providers_length:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetStringTensorElementLength : function (const value:POrtValue; index:size_t; _out:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetStringTensorElement : function (const value:POrtValue; s_len:size_t; index:size_t; s:pointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   FillStringTensorElement : function (value:POrtValue;const s:POrtChar; index:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   AddSessionConfigEntry : function (options:POrtSessionOptions;const config_key:POrtChar;const config_value:POrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateAllocator : function (const session:POrtSession;const mem_info:POrtMemoryInfo; _out:PPOrtAllocator):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ReleaseAllocator : procedure (input:POrtAllocator);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   RunWithBinding : function (session:POrtSession;const run_options:POrtRunOptions;const binding_ptr:POrtIoBinding):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateIoBinding : function (session:POrtSession; _out:PPOrtIoBinding):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseIoBinding : procedure (input:POrtIoBinding);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   BindInput : function (binding_ptr:POrtIoBinding;const name:POrtChar;const val_ptr:POrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   BindOutput : function (binding_ptr:POrtIoBinding;const name:POrtChar;const val_ptr:POrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   BindOutputToDevice : function (binding_ptr:POrtIoBinding;const name:POrtChar;const mem_info_ptr:POrtMemoryInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetBoundOutputNames : function (const binding_ptr:POrtIoBinding; allocator:POrtAllocator; buffer:PPOrtChar; lengths:PPsize_t; count:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetBoundOutputValues : function (const binding_ptr:POrtIoBinding; allocator:POrtAllocator; output:PPPOrtValue; output_count:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ClearBoundInputs : procedure (binding_ptr:POrtIoBinding);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ClearBoundOutputs : procedure (binding_ptr:POrtIoBinding);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   TensorAt : function (value:POrtValue;const location_values:Pint64_t; location_values_count:size_t; _out:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateAndRegisterAllocator : function (env:POrtEnv;const mem_info:POrtMemoryInfo;const arena_cfg:POrtArenaCfg):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetLanguageProjection : function (const ort_env:POrtEnv; projection:OrtLanguageProjection):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionGetProfilingStartTimeNs : function (session:POrtSession; _out:Puint64_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SetGlobalIntraOpNumThreads : function (tp_options:POrtThreadingOptions; intra_op_num_threads:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SetGlobalInterOpNumThreads : function (tp_options:POrtThreadingOptions; inter_op_num_threads:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SetGlobalSpinControl : function (tp_options:POrtThreadingOptions; allow_spinning:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   AddInitializer : function (options:POrtSessionOptions;const name:POrtChar;const val:POrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateEnvWithCustomLoggerAndGlobalThreadPools : function (logging_function:OrtLoggingFunction; logger_param:pointer; log_severity_level:OrtLoggingLevel;const logid:POrtChar;const tp_options:POrtThreadingOptions; _out:PPOrtEnv):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_CUDA : function (options:POrtSessionOptions;const cuda_options:POrtCUDAProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_ROCM : function (options:POrtSessionOptions;const rocm_options:POrtROCMProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_OpenVINO : function (options:POrtSessionOptions;const provider_options:POrtOpenVINOProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetGlobalDenormalAsZero : function (tp_options:POrtThreadingOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateArenaCfg : function (max_mem:size_t; arena_extend_strategy:longint; initial_chunk_size_bytes:longint; max_dead_bytes_per_chunk:longint; _out:PPOrtArenaCfg):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ReleaseArenaCfg : procedure (input:POrtArenaCfg);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   ModelMetadataGetGraphDescription : function (const model_metadata:POrtModelMetadata; allocator:POrtAllocator; value:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_TensorRT : function (options:POrtSessionOptions;const tensorrt_options:POrtTensorRTProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SetCurrentGpuDeviceId : function (device_id:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetCurrentGpuDeviceId : function (device_id:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   KernelInfoGetAttributeArray_float : function (const info:POrtKernelInfo;const name:POrtChar; _out:Psingle; size:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelInfoGetAttributeArray_int64 : function (const info:POrtKernelInfo;const  name:POrtChar; _out:Pint64_t; size:Psize_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateArenaCfgV2 : function (const arena_config_keys:PPOrtChar;const arena_config_values:Psize_t; num_keys:size_t; _out:PPOrtArenaCfg):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   AddRunConfigEntry : function (options:POrtRunOptions;const config_key:POrtChar;const config_value:POrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreatePrepackedWeightsContainer : function (_out:PPOrtPrepackedWeightsContainer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleasePrepackedWeightsContainer : procedure (input:POrtPrepackedWeightsContainer);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateSessionWithPrepackedWeightsContainer : function (const env:POrtEnv;const model_path:PORTCHAR_T;const options:POrtSessionOptions; prepacked_weights_container:POrtPrepackedWeightsContainer; _out:PPOrtSession):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateSessionFromArrayWithPrepackedWeightsContainer : function (const env:POrtEnv;const model_data:pointer; model_data_length:size_t;const options:POrtSessionOptions; prepacked_weights_container:POrtPrepackedWeightsContainer;_out:PPOrtSession):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_TensorRT_V2 : function (options:POrtSessionOptions;const tensorrt_options:POrtTensorRTProviderOptionsV2):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateTensorRTProviderOptions : function (_out:PPOrtTensorRTProviderOptionsV2):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   UpdateTensorRTProviderOptions : function (tensorrt_options:POrtTensorRTProviderOptionsV2;const provider_options_keys:PPOrtChar;const provider_options_values:PPOrtChar; num_keys:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   GetTensorRTProviderOptionsAsString : function (const tensorrt_options:POrtTensorRTProviderOptionsV2; allocator:POrtAllocator; ptr:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseTensorRTProviderOptions : procedure (input:POrtTensorRTProviderOptionsV2);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   EnableOrtCustomOps : function (options:POrtSessionOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   RegisterAllocator : function (env:POrtEnv; allocator:POrtAllocator):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   UnregisterAllocator : function (env:POrtEnv;const mem_info:POrtMemoryInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   IsSparseTensor : function (const value:POrtValue; _out:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateSparseTensorAsOrtValue : function (allocator:POrtAllocator;const dense_shape:Pint64_t; dense_shape_len:size_t; _type:ONNXTensorElementDataType; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   FillSparseTensorCoo : function (ort_value:POrtValue;const data_mem_info:POrtMemoryInfo;const values_shape:Pint64_t; values_shape_len:size_t;const values:pointer;const indices_data:Pint64_t; indices_num:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   FillSparseTensorCsr : function (ort_value:POrtValue;const data_mem_info:POrtMemoryInfo;const values_shape:Pint64_t; values_shape_len:size_t;const values:pointer;const inner_indices_data:Pint64_t; inner_indices_num:size_t;const outer_indices_data:Pint64_t; outer_indices_num:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   FillSparseTensorBlockSparse : function (ort_value:POrtValue;const data_mem_info:POrtMemoryInfo;const values_shape:Pint64_t; values_shape_len:size_t;const values:pointer;const indices_shape_data:Pint64_t; indices_shape_len:size_t;const indices_data:Pint32_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   CreateSparseTensorWithValuesAsOrtValue : function (const info:POrtMemoryInfo; p_data:pointer;const dense_shape:Pint64_t; dense_shape_len:size_t;const values_shape:Pint64_t; values_shape_len:size_t; _type:ONNXTensorElementDataType; _out:PPOrtValue):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   UseCooIndices : function (ort_value:POrtValue; indices_data:Pint64_t; indices_num:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   UseCsrIndices : function (ort_value:POrtValue; inner_data:Pint64_t; inner_num:size_t; outer_data:Pint64_t; outer_num:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   UseBlockSparseIndices : function (ort_value:POrtValue;const indices_shape:Pint64_t; indices_shape_len:size_t; indices_data:Pint32_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetSparseTensorFormat : function (const ort_value:POrtValue; _out:POrtSparseFormat):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetSparseTensorValuesTypeAndShape : function (const ort_value:POrtValue; _out:PPOrtTensorTypeAndShapeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetSparseTensorValues : function (const ort_value:POrtValue;const _out:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetSparseTensorIndicesTypeShape : function (const ort_value:POrtValue; indices_format:OrtSparseIndicesFormat; _out:PPOrtTensorTypeAndShapeInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetSparseTensorIndices : function (const ort_value:POrtValue; indices_format:OrtSparseIndicesFormat; num_indices:Psize_t;const indices:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   HasValue : function (const value:POrtValue; _out:Plongint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   KernelContext_GetGPUComputeStream : function (const context:POrtKernelContext; _out:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetTensorMemoryInfo : function (const value:POrtValue;const mem_info:PPOrtMemoryInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetExecutionProviderApi : function (const provider_name:POrtChar; version:uint32_t;const provider_api:Ppointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   SessionOptionsSetCustomCreateThreadFn : function (options:POrtSessionOptions; ort_custom_create_thread_fn:OrtCustomCreateThreadFn):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsSetCustomThreadCreationOptions : function (options:POrtSessionOptions; ort_custom_thread_creation_options:pointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsSetCustomJoinThreadFn : function (options:POrtSessionOptions; ort_custom_join_thread_fn:OrtCustomJoinThreadFn):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SetGlobalCustomCreateThreadFn : function (tp_options:POrtThreadingOptions; ort_custom_create_thread_fn:OrtCustomCreateThreadFn):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SetGlobalCustomThreadCreationOptions : function (tp_options:POrtThreadingOptions; ort_custom_thread_creation_options:pointer):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SetGlobalCustomJoinThreadFn : function (tp_options:POrtThreadingOptions; ort_custom_join_thread_fn:OrtCustomJoinThreadFn):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SynchronizeBoundInputs : function (binding_ptr:POrtIoBinding):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SynchronizeBoundOutputs : function (binding_ptr:POrtIoBinding):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_CUDA_V2 : function (options:POrtSessionOptions;const cuda_options:POrtCUDAProviderOptionsV2):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateCUDAProviderOptions : function (_out:PPOrtCUDAProviderOptionsV2):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   UpdateCUDAProviderOptions : function (cuda_options:POrtCUDAProviderOptionsV2;const provider_options_keys:PPOrtChar;const provider_options_values:PPOrtChar; num_keys:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetCUDAProviderOptionsAsString : function (const cuda_options:POrtCUDAProviderOptionsV2; allocator:POrtAllocator; ptr:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseCUDAProviderOptions : procedure (input:POrtCUDAProviderOptionsV2);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_MIGraphX : function (options:POrtSessionOptions;const migraphx_options:POrtMIGraphXProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};

   AddExternalInitializers : function (options:POrtSessionOptions;const initializer_names:PPOrtChar;const initializers:PPOrtValue; initializers_num:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateOpAttr : function (const name:POrtChar;const data:pointer; len:longint; _type:OrtOpAttrType; op_attr:PPOrtOpAttr):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseOpAttr : procedure (input:POrtOpAttr);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateOp : function (const info:POrtKernelInfo;const op_name:POrtChar;const domain:POrtChar; version:longint;const type_constraint_names:PPOrtChar;const type_constraint_values:PONNXTensorElementDataType;const type_constraint_count:longint;const attr_values:PPOrtOpAttr; attr_count:longint; input_count:longint;  output_count:longint; ort_op:PPOrtOp):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   InvokeOp : function (const context:POrtKernelContext;const ort_op:POrtOp;const input_values:PPOrtValue; input_count:longint;const output_values:PPOrtValue; output_count:longint):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseOp : procedure (input:POrtOp);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider : function (options:POrtSessionOptions;const provider_name:POrtChar;const provider_options_keys:PPOrtChar;const provider_options_values:PPOrtChar; num_keys:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CopyKernelInfo : function (const info:POrtKernelInfo; info_copy:PPOrtKernelInfo):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseKernelInfo : procedure (input:POrtKernelInfo);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetTrainingApi : function (version:uint32_t):POrtTrainingApi;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   SessionOptionsAppendExecutionProvider_CANN : function (options:POrtSessionOptions;const cann_options:POrtCANNProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   CreateCANNProviderOptions : function (_out:PPOrtCANNProviderOptions):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   UpdateCANNProviderOptions : function (cann_options:POrtCANNProviderOptions;const provider_options_keys:PPOrtChar;const provider_options_values:PPOrtChar; num_keys:size_t):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   GetCANNProviderOptionsAsString : function (const cann_options:POrtCANNProviderOptions; allocator:POrtAllocator; ptr:PPOrtChar):POrtStatus;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   ReleaseCANNProviderOptions : procedure (input:POrtCANNProviderOptions);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   MemoryInfoGetDeviceType : procedure (const ptr : POrtMemoryInfo ; _out : POrtMemoryInfoDeviceType); {$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   UpdateEnvWithCustomLogLevel : function(const env:POrtEnv; log_level: OrtLoggingLevel):POrtStatus; {$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
 end;

 OrtCustomOpInputOutputCharacteristic = (INPUT_OUTPUT_REQUIRED = 0,INPUT_OUTPUT_OPTIONAL);
 OrtCustomOp = record
     version : uint32_t;
     CreateKernel : function (const op:POrtCustomOp;const api:POrtApi;const info:POrtKernelInfo):pointer;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetName : function (const op:POrtCustomOp):POrtChar;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetExecutionProviderType : function (const op:POrtCustomOp):POrtChar;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetInputType : function (const op:POrtCustomOp; index:size_t):ONNXTensorElementDataType;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetInputTypeCount : function (const op:POrtCustomOp):size_t;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetOutputType : function (const op:POrtCustomOp; index:size_t):ONNXTensorElementDataType;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetOutputTypeCount : function (const op:POrtCustomOp):size_t;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     KernelCompute : procedure (op_kernel:pointer; context:POrtKernelContext);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     KernelDestroy : procedure (op_kernel:pointer);{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetInputCharacteristic : function (const op:POrtCustomOp; index:size_t):OrtCustomOpInputOutputCharacteristic;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
     GetOutputCharacteristic : function (const op:POrtCustomOp; index:size_t):OrtCustomOpInputOutputCharacteristic;{$ifdef FPC}ORT_API_CALL{$else}stdcall{$endif};
   end;

(* Const before type ignored *)

  const DefaultLanguageProjection = OrtLanguageProjection.ORT_PROJECTION_CPLUSPLUS;

var
  Global : POrtApiBase;
  Api: POrtApi;

  function OrtGetApiBase:POrtApiBase;cdecl;  external dllname;

(**
 * \param use_arena zero: false. non-zero: true.
 *)

  function OrtSessionOptionsAppendExecutionProvider_CPU(options: POrtSessionOptions; use_arena: longint):POrtStatus;cdecl;       external dllname;

  function OrtSessionOptionsAppendExecutionProvider_CUDA(options: POrtSessionOptions; device_id: longint):POrtStatus;cdecl;      external dllname;

  function OrtSessionOptionsAppendExecutionProvider_MIGraphX(options: POrtSessionOptions; device_id: longint):POrtStatus;cdecl;  external dllname;
  //
  function OrtSessionOptionsAppendExecutionProvider_Tensorrt(options: POrtSessionOptions; device_id: longint):POrtStatus;cdecl;  external dllname;
(**
 * [[deprecated]]
 * This export is deprecated.
 * The OrtSessionOptionsAppendExecutionProvider_DML export on the OrtDmlApi should be used instead.
 *
 * Creates a DirectML Execution Provider which executes on the hardware adapter with the given device_id, also known as
 * the adapter index. The device ID corresponds to the enumeration order of hardware adapters as given by
 * IDXGIFactory::EnumAdapters. A device_id of 0 always corresponds to the default adapter, which is typically the
 * primary display GPU installed on the system. A negative device_id is invalid.
 *)
  function OrtSessionOptionsAppendExecutionProvider_DML(const options: POrtSessionOptions ;const  device_id:longint):POrtStatus; cdecl;external dllname;

  (**
 * [[deprecated]]
 * This export is deprecated.
 * The OrtSessionOptionsAppendExecutionProvider_DML1 export on the OrtDmlApi should be used instead.
 *
 * Creates a DirectML Execution Provider using the given DirectML device, and which executes work on the supplied D3D12
 * command queue. The DirectML device and D3D12 command queue must have the same parent ID3D12Device, or an error will
 * be returned. The D3D12 command queue must be of type DIRECT or COMPUTE (see D3D12_COMMAND_LIST_TYPE). If this
 * function succeeds, the inference session maintains a strong reference on both the dml_device and the command_queue
 * objects.
 * See also: DMLCreateDevice
 * See also: ID3D12Device::CreateCommandQueue
 *)
function OrtSessionOptionsAppendExecutionProviderEx_DML(
  const options:POrtSessionOptions; dml_device:Pointer {IDMLDevice*};
  const cmd_queue:Pointer {ID3D12CommandQueue*}):POrtStatus;   stdcall;external dllname;


implementation

{ OrtROCMProviderOptions }

class operator OrtROCMProviderOptions.Initialize({$ifdef fpc}var{$else}out{$endif}  dest: OrtROCMProviderOptions);
begin
  dest.gpu_mem_limit:=SIZE_MAX;
  dest.do_copy_in_default_stream:=1;
end;

{ OrtCUDAProviderOptions }

class operator OrtCUDAProviderOptions.Initialize({$ifdef fpc}var{$else}out{$endif} dest: OrtCUDAProviderOptions);
begin
  dest.gpu_mem_limit:=SIZE_MAX;
  dest.do_copy_in_default_stream:=1;
end;

//var s:POrtChar;
initialization
  Global:=OrtGetApiBase;
  Api:=Global.GetApi(ORT_API_VERSION);
  if not Assigned(Api) and isConsole then begin
    writeln('Cannot load ONNXRuntime API, possibly wrong version');
    writeln(' - requested version [',ORT_API_VERSION,']');
    writeLn(' - CurrentVersion[',global.GetVersionString,']');
  end;

end.
