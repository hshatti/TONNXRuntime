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

(*
  This is a reworked/refactored implementation of the
  original C++ onnxruntime library (link checked as of Sep-2022):
  https://github.com/microsoft/onnxruntime/tree/main/include/onnxruntime/core/session

  the problem seems to be with the original C++ code is that it uses a
  trivial implementation of management operators to track the library pointers,
  but it can accindenly dispose them while they are still in scope, because the
  <Base> implementation handsover the ownership of these pointers on every
  assignment, and clears it self instead of keeping track of each reference made
  or gone out of scope, I would choose to do the same thing in pascal but since
  pascal is more well behaved I decided to implement a simple global
  housekeeper that keeps everything in tracked, if any pointer
  goes out of scope, the owned reference will be managed and freed only if the
  pointer is completely no longer used without compromizing the library
  performance.




*)

unit onnxruntime;
{.$define MEM_DEBUG}  // debug memory housekeeping mechanism
{$ifdef MEM_DEBUG}
{$APPTYPE CONSOLE}
{$ENDIF}

{$H+}
{$ifdef fpc}
  {$MACRO ON}
  {$mode delphi}
  {$ModeSwitch advancedrecords}
  {$ModeSwitch typehelpers}
  {$define outvar:=var}
  {$PointerMath on}
{$else}
  {$define outvar:=out}
{$endif}
// NO_HASHMAP will use internal ValueKey implementation, disabeling in Delphi  may result unwanted behaviour
{$define NO_HASHMAP}


{.$define NO_SMARTPTR}
//{$define ORT_NO_EXCEPTION}

interface

uses
  SysUtils, TypInfo, onnxruntime_pas_api, Generics.Collections, Generics.Defaults{$ifndef fpc}, SyncObjs{$endif};

type ortstring = ansistring;


const ORTTensorTypes: array[ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED..ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16] of ortstring =
  (
    'undefined', 'float', 'uint8', 'int8', 'uint16', 'int16', 'int32', 'int64', 'string',
   'bool', 'float16', 'double', 'uint32', 'uint64', 'complex64', 'complex128', 'bfloat16'
   );
const DEFAULT_LOGID :PAnsiChar= 'ONNXRuntimePascal';

type
  Tinfo =record
    kind:TTypeKind;
    name:shortstring
  end;

  uint16_t=UInt16;

  TCompareFunc<T>=function (const a,b:T):integer;

  { TTools }

  TTools=class
    class function min<T>(const a,b:T):T; static;
    class function max<T>(const a,b:T):T; static;
    class function IfThen<T>(const cond:boolean;const whenTrue,whenFalse:T):T; static;
    class procedure QuickSort<T>(var Arr: array of T; L, R: Longint; const Descending: boolean; const Compare: TComparefunc<T>); static;
    class function BinSearch<T>(const Arr:array of T;const Val:T; R:integer):integer; static;
    class function cmp<T>(const a, b: T): integer; static;
    class function reverse<T>(const a:TArray<T>):TArray<T>;
  end;

  {$ifndef NO_SMARTPTR}
  PORTAllocatedFree = ^TORTAllocatedFree;
  { TORTAllocatedFree }
  // Light functor to release memory with OrtAllocator
  TORTAllocatedFree = record
    allocator_:POrtAllocator;
    constructor Create(const allocator:POrtAllocator);
    procedure call(var ptr:pointer);
  end;

  { TSmartPtr }

  TSmartPtr<T> = record
    Type PT=^T;
    var
    // similar as overloading [] operators for property x[v: string]: integer read gx write sx; default;
    Instance: PT ; // default keyword for non property.
    RefCount: PLongint;
    procedure DecRef();

    class operator Initialize({$ifdef fpc}var{$else}out{$endif} dst: TSmartPtr<T>);
    class operator Finalize(var dst: TSmartPtr<T>);
    {$ifdef fpc}
    class operator AddRef(outvar src: TSmartPtr<T>);
    class operator Copy(constref src: TSmartPtr<T>; outvar dst: TSmartPtr<T>);
    {$else}
    class operator Assign(var dst: TSmartPtr<T>;const [ref] src: TSmartPtr<T>);
    {$endif}


    // implicit or explicit operator should be used before "default" field
    class operator Implicit(const src: PT): TSmartPtr<T>;
    //class operator Implicit(const src: TSmartPtr<T>):T;
    procedure Assign(const val: PT);
  end;

  { TSmartPtr }
  // P is a disposer record/class with one "call" functor at least e.g:
  // (TDisposer = record procedure Call(var value:T) end;)
  TSmartPtr<T,P> = record
  type PT=^T;
  var
    // similar as overloading [] operators for property x[v: string]: integer read gx write sx; default;
    Instance: PT ; // default keyword for non property.
    RefCount: PLongint;
    DisposerFunc: TORTAllocatedFree;
    constructor Create(const val:PT;const aDisposer:TORTAllocatedFree);
    procedure DecRef();

    class operator Initialize({$ifdef fpc}var{$else}out{$endif} dst: TSmartPtr<T,P>);
    class operator Finalize(var dst: TSmartPtr<T,P>);
    {$ifdef fpc}
    class operator AddRef(var src: TSmartPtr<T,P>);
    class operator Copy(constref src: TSmartPtr<T,P>;var dst: TSmartPtr<T,P>);
    {$else}
    class operator Assign(var dst: TSmartPtr<T,P>; const [ref] src: TSmartPtr<T,P>);
    {$endif}

    // implicit or explicit operator should be used before "default" field
    class operator Implicit(const src: PT): TSmartPtr<T,P>;
    //class operator Implicit(const src: TSmartPtr<T,P>):T;
    procedure Assign(const val:PT);
  end;
  {$endif}
  // You may want to use generic class based hashmaps or dictionaries, TOrderedKeyValues below should keep everything in the stack

  {$ifdef NO_HASHMAP}
  { TOrderedKeyValueList }

  TOrderedKeyValueList<TK,TV>=record
  type
    TKeyArray = TArray<TK>;
    TValueArray =TArray<TV>;
    TSearcher = function (const arr:array of TK;const val:TK;R:integer):integer;
  public
    Keys:TKeyArray;
    Values:TValueArray;
  private
    FindKey:TSearcher ;
    function GetValues( key: TK): TV;
    procedure SetValues( key: TK; AValue: TV);
  public
    class function Create():TOrderedKeyValueList<TK,TV>; static;
    function ContainsKey(const key: TK): boolean;
    function IndexOfKey(const Key:TK):integer;
    function AddOrSetValue(const Key:TK;const AValue:TV):integer;
    procedure Add(const Key:TK;const AValue:TV);
    function TryGetValue(const Key:TK;out Value:TV):boolean;
    property Items[key:TK]:TV read GetValues write SetValues ;default;
    function Count:integer;
    procedure RemoveIndex(const index:integer);
    procedure Remove(const key:TK);
    class operator initialize({$ifdef fpc}outvar{$else}out{$endif} v:TOrderedKeyValueList<TK,TV>);
    class operator Finalize(var dst:TOrderedKeyValueList<TK,TV>);

  end;
  PMemHouseKeeper = ^TMemHouseKeeper;
  TMemHouseKeeper = TOrderedKeyValueList<Pointer,PLongInt>;

  {$endif}

  AllocatedStringPtr={$ifdef NO_SMARTPTR}ortstring{$else}TSmartPtr<OrtChar,TORTAllocatedFree>{$endif};

  { OrtException }

  OrtException = class(Exception)

    code_ :OrtErrorCode;
    property GetErrorCode:OrtErrorCode read code_;
    function what :ortstring;
    constructor Create(const str:ortstring; code :OrtErrorCode);
  end;

  { Float16_t }

  Float16_t=record
    value:uint16_t;
    constructor Create(const v:uint16_t);
    class operator Implicit(const v:Float16_t):uint16_t;overload;
    class operator Implicit(const v:uint16_t):Float16_t;overload;
  end;

  { BFloat16_t }

  BFloat16_t=record
    value:uint16_t;
    constructor Create(const v:uint16_t);
    class operator Implicit(const v:BFloat16_t):uint16_t;overload;
    class operator Implicit(const v:uint16_t):BFloat16_t;overload;
  end;

  { TORTBase }

  TORTBase<T>=record
  type
    PT=^T;
  private
    procedure NewRef();
    procedure Assign(const val: Pointer);
    procedure DecRef();
    //function RefCount():PLongInt;
  public
    p_:Pointer; // don't change to PT, because of later implementation of OrtRelease which will give compiler error;
    function release:PT;
    procedure OrtRelease();
    constructor Create(const v:PT);
    class operator Initialize({$ifdef fpc}outvar{$else}out{$endif} v:TORTBase<T>);
    class operator Finalize(var v:TORTBase<T>);
    {$ifdef fpc}
    class operator Copy(constref src: TORTBase<T>;var dst:TORTBase<T>);
    class operator AddRef(var src:TORTBase<T>);
    {$else}
    class operator Assign(var dst: TORTBase<T>;const [ref] src:TORTBase<T>);
    {$endif}
    //class operator Implicit(const v: TORTBase<T>):PT;
    class operator Implicit(const v: PT):TORTBase<T>;
    class operator Equal(const a,b:TORTBase<T>):boolean;
  end;


  //PORTEnv                    = ^TORTEnv                    ;
  //PORTOrtCustomOpDomain      = ^TORTCustomOpDomain         ;
  //PORTRunOptions             = ^TORTRunOptions             ;
  //PORTSessionOptions         = ^TORTSessionOptions         ;
  //PORTModelMetadata          = ^TORTModelMetadata          ;
  //PORTSession                = ^TORTSession                ;
  //PORTORTTensorTypeAndShapeInfo = ^TORTTensorTypeAndShapeInfo ;
  //PORTSequenceTypeInfo       = ^TORTSequenceTypeInfo       ;
  //PORTMapTypeInfo            = ^TORTMapTypeInfo            ;
  //PORTTypeInfo               = ^TORTTypeInfo               ;
  //PORTORTValue               = ^TORTValue                  ;
  //PORTOrtMemoryInfo          = ^TORTMemoryInfo             ;
  //PORTOrtAllocator           = ^TORTAllocator              ;
  //PORTIoBinding              = ^TORTIoBinding              ;

  TORTEnv                    = TORTBase<OrtEnv>;
  TORTValue                  = TORTBase<OrtValue>;
  TORTIoBinding              = TORTBase<OrtIoBinding>  ;
  TORTTypeInfo               = TORTBase<OrtTypeInfo>;
  TORTCustomOpDomain         = TORTBase<OrtCustomOpDomain>;
  TORTRunOptions             = TORTBase<OrtRunOptions>;
  TORTSessionOptions         = TORTBase<OrtSessionOptions>;
  TORTModelMetadata          = TORTBase<OrtModelMetadata> ;
  TORTSession                = TORTBase<OrtSession>;
  TORTMapTypeInfo            = TORTBase<OrtMapTypeInfo>;
  TORTSequenceTypeInfo       = TORTBase<OrtSequenceTypeInfo>;
  TORTAllocator              = TORTBase<OrtAllocator> ;
  TORTMemoryInfo             = TORTBase<OrtMemoryInfo> ;
  TORTArenaCfg               = TORTBase<OrtArenaCfg> ;
  TORTTensorTypeAndShapeInfo = TORTBase<OrtTensorTypeAndShapeInfo>;
  { TOrtTensor }

  TORTTensor<T> = record
  Type PT=^T;
  private
    FData:TArray<T>;
    FShape:TArray<int64_t>;
    FSize:Size_t;
    function Getindex1(x: int64_t): T;
    function Getindex3( x,y,z: int64_t): T;
    function Getindex2(x,y: int64_t): T;
    function GetIndex4(x,y,z,w: int64_t): T;
    procedure Setindex1(x: int64_t; AValue: T);
    procedure Setindex3(x,y,z: int64_t; AValue: T);
    procedure Setindex2(x,y: int64_t; AValue: T);
    procedure SetIndex4(x,y,z,w: int64_t; AValue: T);
    procedure SetShape(AValue: TArray<int64_t>);
  public
    function ElementCount:size_t;
    function ByteCount:size_t;
    function DimensionCount:size_t;
    function Idx(const x,y,z:Int64_t):int64_t;
    constructor Create(const AShape:TArray<int64_t>);
    property Shape:TArray<int64_t> read FShape write SetShape;
    property Index4[x,y,z,w:int64_t]:T read GetIndex4 write SetIndex4;
    property Index3[x,y,z:int64_t]:T read Getindex3 write Setindex3 ; default;
    property index2[x,y:int64_t]:T read Getindex2 write Setindex2  ;
    property index1[x:int64_t]:T read GetIndex1 write SetIndex1;
    {$ifdef fpc}
    function ToString(const Separator:ortstring=', '):ortstring;
    {$endif}
    function ToValue:TORTValue;
    class function FromValue(const val:TORTValue;const copyData:boolean=true):TORTTensor<T>;static;
    class operator Implicit(const aValue:TORTValue):TORTTensor<T>;
    class operator Implicit(const aValue:TORTTensor<T>):TORTValue;
  end;


  {$ifdef MEM_TABLE}
  THKHelp = record helper for TMemHouseKeeper
    procedure print();
  end;
  {$endif}

  {$ifndef NO_HASHMAP}

  TORTNameValue<TKey,TValue> =  class(TDictionary<TKey,TValue>)
  private
    function GetKeys(idx: size_t): TKey;
    function GetValues(idx: Size_t): TValue;
  public
    property Values[idx:Size_t]:TValue read GetValues;
    property Keys[idx:size_t]:TKey read GetKeys;
  end;
  TNameValueList = TORTNameValue<ansistring,TORTValue>;
  TORTProviderOptions= TDictionary<ortstring,ortstring>;

  {$else}
//  TMemHousekeeper = TOrderedKeyValueList<Pointer,Plongint>;
  {$ifdef MEM_TABLE}
  {$endif}
  TORTNameValueList = TOrderedKeyValueList<ortstring,TORTValue>;
  TORTProviderOptions=TOrderedKeyValueList<ortstring,ortstring>;
  {$endif}

  { TEnvHelper }

  TORTEnvHelper = record helper for TORTEnv
    class function Create(logging_level:OrtLoggingLevel {= ORT_LOGGING_LEVEL_WARNING}; const logid: POrtChar  = nil):TORTEnv;                                                                                                       overload;static;
    class function Create(logging_level :OrtLoggingLevel ; const logid :POrtChar ; logging_function:OrtLoggingFunction ; logger_param: Pointer ):TORTEnv;                                                                          overload; static;
    class function Create(const tp_options :POrtThreadingOptions ; logging_level: OrtLoggingLevel = ORT_LOGGING_LEVEL_WARNING; const logid: POrtChar  = nil):TORTEnv;                                                               overload;static;
    class function Create(const tp_options :POrtThreadingOptions ; logging_function: OrtLoggingFunction ; logger_param:Pointer; logging_level :OrtLoggingLevel  = ORT_LOGGING_LEVEL_WARNING; const logid:POrtChar  = nil):TORTEnv;  overload;static;
    function  EnableTelemetryEvents():TORTEnv;inline;   ///< Wraps OrtApi::EnableTelemetryEvents
    function DisableTelemetryEvents():TORTEnv;inline;  ///< Wraps OrtApi::DisableTelemetryEvents
    function CreateAndRegisterAllocator(const mem_info :TOrtMemoryInfo; const arena_cfg:TOrtArenaCfg):TORTEnv;inline;  ///< Wraps OrtApi::CreateAndRegisterAllocator
    function UpdateLogLevel(const log_level:OrtLoggingLevel):TORTEnv;
  end;



  { TCustomOpDomainHelper }

  TORTCustomOpDomainHelper = record helper for TORTCustomOpDomain
    /// \brief Wraps OrtApi::CreateCustomOpDomain
    class function Create(const domain: POrtChar):TORTCustomOpDomain; static;
    procedure Add( op:POrtCustomOp);  ///< Wraps CustomOpDomain_Add
  end;



  { TRunOptionsHelper }

  TORTRunOptionsHelper = record helper for TORTRunOptions
     // RunOption() created in TORTBase intializer    ///< Wraps OrtApi::CreateRunOptions

     function SetRunLogVerbosityLevel(level:longint):TORTRunOptions;  ///< Wraps OrtApi::RunOptionsSetRunLogVerbosityLevel
     function GetRunLogVerbosityLevel():longint;      ///< Wraps OrtApi::RunOptionsGetRunLogVerbosityLevel
     //
     function SetRunLogSeverityLevel(level:longint):TORTRunOptions;  ///< Wraps OrtApi::RunOptionsSetRunLogSeverityLevel
     function GetRunLogSeverityLevel():longint ;       ///< Wraps OrtApi::RunOptionsGetRunLogSeverityLevel
     //
     function SetRunTag(const run_tag:POrtChar):TORTRunOptions;  ///< wraps OrtApi::RunOptionsSetRunTag
     function GetRunTag():POrtChar;               ///< Wraps OrtApi::RunOptionsGetRunTag
     //
     function AddConfigEntry(const config_key:POrtChar; const config_value:POrtChar):TORTRunOptions;  ///< Wraps OrtApi::AddRunConfigEntry
     //
     ///** \brief Terminates all currently executing Session::Run calls that were made using this RunOptions instance
     // *
     // * If a currently executing session needs to be force terminated, this can be called from another thread to force it to fail with an error
     // * Wraps OrtApi::RunOptionsSetTerminate
     // */
     function SetTerminate():TORTRunOptions;
     //
     ///** \brief Clears the terminate flag so this RunOptions instance can be used in a new Session::Run call without it instantly terminating
     // *
     // * Wraps OrtApi::RunOptionsUnsetTerminate
     // */
     function UnsetTerminate():TORTRunOptions;

  end;


  { TSessionOptionsHelper }

  TORTSessionOptionsHelper = record helper for TORTSessionOptions
     function Clone():TORTSessionOptions;  ///< Creates and returns a copy of this SessionOptions object. Wraps OrtApi::CloneSessionOptions
     function SetIntraOpNumThreads(intra_op_num_threads :longint):TORTSessionOptions;                              ///< Wraps OrtApi::SetIntraOpNumThreads
     function SetInterOpNumThreads(inter_op_num_threads :longint):TORTSessionOptions;                              ///< Wraps OrtApi::SetInterOpNumThreads
     function SetGraphOptimizationLevel(graph_optimization_level :GraphOptimizationLevel):TORTSessionOptions;  ///< Wraps OrtApi::SetSessionGraphOptimizationLevel
     function EnableCpuMemArena():TORTSessionOptions;   ///< Wraps OrtApi::EnableCpuMemArena
     function DisableCpuMemArena():TORTSessionOptions;  ///< Wraps OrtApi::DisableCpuMemArena
     function SetOptimizedModelFilePath(const optimized_model_filepath: PORTCHAR_T): TORTSessionOptions;  ///< Wraps OrtApi::SetOptimizedModelFilePath
     function EnableProfiling(const profile_file_prefix:PORTCHAR_T):TORTSessionOptions;  ///< Wraps OrtApi::EnableProfiling
     function DisableProfiling():TORTSessionOptions;                                     ///< Wraps OrtApi::DisableProfiling
     function EnableOrtCustomOps():TORTSessionOptions;  ///< Wraps OrtApi::EnableOrtCustomOps
     function EnableMemPattern():TORTSessionOptions;   ///< Wraps OrtApi::EnableMemPattern
     function DisableMemPattern():TORTSessionOptions;  ///< Wraps OrtApi::DisableMemPattern
     function SetExecutionMode(execution_mode:ExecutionMode):TORTSessionOptions;  ///< Wraps OrtApi::SetSessionExecutionMode
     function SetLogId(const logid:POrtChar):TORTSessionOptions;     ///< Wraps OrtApi::SetSessionLogId
     function SetLogSeverityLevel(level:Longint):TORTSessionOptions;  ///< Wraps OrtApi::SetSessionLogSeverityLevel
     function Add(custom_op_domain:POrtCustomOpDomain):TORTSessionOptions;  ///< Wraps OrtApi::AddCustomOpDomain
     function DisablePerSessionThreads():TORTSessionOptions;  ///< Wraps OrtApi::DisablePerSessionThreads
     function AddConfigEntry(const config_key:POrtChar; const config_value:POrtChar):TORTSessionOptions;                                      ///< Wraps OrtApi::AddSessionConfigEntry
     function AddInitializer(const name:POrtChar; const ort_val:POrtValue):TORTSessionOptions;                                             ///< Wraps OrtApi::AddInitializer
     function AddExternalInitializers(const names: TArray<ortstring>; const ort_values: array of TORTValue): TORTSessionOptions;  ///< Wraps OrtApi::AddExternalInitializers
     function AppendExecutionProvider_CUDA(const provider_options:OrtCUDAProviderOptions):TORTSessionOptions;               ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA
     function AppendExecutionProvider_CUDA_V2(const provider_options:OrtCUDAProviderOptionsV2):TORTSessionOptions;          ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA_V2
     function AppendExecutionProvider_ROCM(const provider_options:OrtROCMProviderOptions):TORTSessionOptions;               ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_ROCM
     function AppendExecutionProvider_OpenVINO(const provider_options:OrtOpenVINOProviderOptions):TORTSessionOptions;       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO
     function AppendExecutionProvider_TensorRT(const provider_options:OrtTensorRTProviderOptions):TORTSessionOptions;       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
     function AppendExecutionProvider_TensorRT_V2(const provider_options:OrtTensorRTProviderOptionsV2):TORTSessionOptions;  ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
     function AppendExecutionProvider_MIGraphX(const provider_options:OrtMIGraphXProviderOptions):TORTSessionOptions;       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX
     ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CANN
     function AppendExecutionProvider_CANN(const provider_options:OrtCANNProviderOptions):TORTSessionOptions;
     /// Wraps OrtApi::SessionOptionsAppendExecutionProvider. Currently supports SNPE and XNNPACK.
     function AppendExecutionProvider(const provider_name:ortstring; const provider_options:TORTProviderOptions):TORTSessionOptions;
     function SetCustomCreateThreadFn( ort_custom_create_thread_fn:OrtCustomCreateThreadFn):TORTSessionOptions;  ///< Wraps OrtApi::SessionOptionsSetCustomCreateThreadFn
     function SetCustomThreadCreationOptions(ort_custom_thread_creation_options:Pointer):TORTSessionOptions;      ///< Wraps OrtApi::SessionOptionsSetCustomThreadCreationOptions
     function SetCustomJoinThreadFn( ort_custom_join_thread_fn:OrtCustomJoinThreadFn):TORTSessionOptions;        ///< Wraps OrtApi::SessionOptionsSetCustomJoinThreadFn

  end;

  { TModelMetadataHelper }

  TORTModelMetadataHelper = record helper for TORTModelMetadata
    function GetProducerNameAllocated(allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::ModelMetadataGetProducerName
    function GetGraphNameAllocated(allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::ModelMetadataGetGraphName
    function GetDomainAllocated(allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::ModelMetadataGetDomain
    function GetDescriptionAllocated(allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::ModelMetadataGetDescription
    function GetGraphDescriptionAllocated(allocator: TOrtAllocator
      ): AllocatedStringPtr;  ///< Wraps OrtApi::ModelMetadataGetGraphDescription
    function GetCustomMetadataMapKeysAllocated(allocator: TOrtAllocator): TArray<
      AllocatedStringPtr>;  ///< Wraps OrtApi::ModelMetadataGetCustomMetadataMapKeys
    function LookupCustomMetadataMapAllocated(const key: POrtChar;
      allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::ModelMetadataLookupCustomMetadataMap

    function GetVersion():int64_t ;  ///< Wraps OrtApi::ModelMetadataGetVersion


  end;
  { TIoBindingHelper }

  TORTIoBindingHelper = record helper for TORTIoBinding
    class function IoBinding(var session:TORTSession):TORTIoBinding;  static;
    procedure BindInput(const name: POrtChar; const value: TORTValue);
    procedure BindOutput(const name: POrtChar; const value: TORTValue); overload;
    procedure BindOutput(const name: POrtChar; const mem_info: TORTMemoryInfo); overload;
    function GetOutputNames()                    :TArray<ortstring>;overload;
    function GetOutputNames(var alloc:TORTAllocator):TArray<ortstring>;overload;
    function GetOutputValues()                   :TArray<TORTValue>;overload;
    function GetOutputValues(var alloc:TORTAllocator):TArray<TORTValue>;overload;
    procedure ClearBoundInputs();
    procedure ClearBoundOutputs();
    procedure SynchronizeInputs();
    procedure SynchronizeOutputs();
  private
    function GetOutputNamesHelper(allocator: POrtAllocator) :TArray<ortstring>; // result must must release when out of scope
    function GetOutputValuesHelper(allocator: POrtAllocator):TArray<TORTValue>; // result must must release when out of scope
  end;


  { TSessionHelper }

  TORTSessionHelper = record helper for TORTSession
                                                                                                      ///< Create an empty Session object, must be assigned a valid one to be used
    class function Create(const env:TORTEnv ; const  model_path:PORTCHAR_T; const options:TORTSessionOptions):TORTSession;                                                           overload;static;  ///< Wraps OrtApi::CreateSession
    class function Create(const env:TORTEnv ; const  model_path:PORTCHAR_T; const options:TORTSessionOptions; prepacked_weights_container:POrtPrepackedWeightsContainer):TORTSession;overload;static;  ///< Wraps OrtApi::CreateSessionWithPrepackedWeightsContainer
    class function Create(const env:TORTEnv ; const  model_data:Pointer; model_data_length:size_t; const options:TORTSessionOptions):TORTSession;                                    overload;static;    ///< Wraps OrtApi::CreateSessionFromArray
    class function Create(const env:TORTEnv ; const  model_data:Pointer; model_data_length:size_t; const options:TORTSessionOptions;
             prepacked_weights_container:POrtPrepackedWeightsContainer):TORTSession; overload;static;  ///< Wraps OrtApi::CreateSessionFromArrayWithPrepackedWeightsContainer
    class function Create(const model_path:TFileName):TORTSession;            overload;static;
    (** \brief Run the model returning results in an Ort allocated vector.
     *
     * Wraps OrtApi::Run
     *
     * The caller provides a list of inputs and a list of the desired outputs to return.
     *
     * See the output logs for more information on warnings/errors that occur while processing the model.
     * Common errors are.. (TODO)
     *
     * \param[in] run_options
     * \param[in] input_names Array of null terminated strings of length input_count that is the list of input names
     * \param[in] input_values Array of Value objects of length input_count that is the list of input values
     * \param[in] input_count Number of inputs (the size of the input_names & input_values arrays)
     * \param[in] output_names Array of C style strings of length output_count that is the list of output names
     * \param[in] output_count Number of outputs (the size of the output_names array)
     * \return A std::vector of Value objects that directly maps to the output_count (eg. output_name[0] is the first entry of the returned vector)
     *)
     function Run(const run_options:TORTRunOptions; const Inputs: TORTNameValueList):TORTNameValueList; overload;
     function Run(const run_options:TORTRunOptions; const Inputs: TORTNameValueList;  Allocator: TOrtAllocator): TORTNameValueList; overload;
     function Run(const run_options:TORTRunOptions; const input_names :PPOrtChar ; const input_values:PORTValue;  input_count:size_t; const output_names:PPOrtChar ; output_names_count:size_t ):TArray<TORTValue>;                            overload;
     function Run(const Inputs:TORTNameValueList):TORTNameValueList; overload;
    (** \brief Run the model returning results in user provided outputs
     * Same as Run(const RunOptions&, const char* const*, const Value*, size_t,const char* const*, size_t)
     *)
    procedure Run(const run_options:TORTRunOptions; const input_names:PPOrtChar; const input_values:PORTValue; input_count:size_t;
             const output_names:PPOrtChar; output_values:PORTValue; output_count:size_t);                                           overload;

    procedure Run(const run_options:TORTRunOptions; const io_binding:TORTIoBinding);  overload;///< Wraps OrtApi::RunWithBinding

    function GetInputCount():size_t ;                   ///< Returns the number of model inputs
    function GetOutputCount():size_t ;                  ///< Returns the number of model outputs
    function GetOverridableInitializerCount() :size_t;  ///< Returns the number of inputs that have defaults that can be overridden

    (** \brief Returns a copy of input name at the specified index.
     *
     * \param index must less than the value returned by GetInputCount()
     * \param allocator to allocate memory for the copy of the name returned
     * \return a instance of smart pointer that would deallocate the buffer when out of scope.
     *  The OrtAllocator instances must be valid at the point of memory release.
     *)
    function GetInputNameAllocated(index: size_t; allocator: TOrtAllocator
      ): AllocatedStringPtr;

    (** \brief Returns a copy of output name at then specified index.
     *
     * \param index must less than the value returned by GetOutputCount()
     * \param allocator to allocate memory for the copy of the name returned
     * \return a instance of smart pointer that would deallocate the buffer when out of scope.
     *  The OrtAllocator instances must be valid at the point of memory release.
     *)
    function GetOutputNameAllocated(index: size_t; allocator: TOrtAllocator
      ): AllocatedStringPtr;

    (** \brief Returns a copy of the overridable initializer name at then specified index.
     *
     * \param index must less than the value returned by GetOverridableInitializerCount()
     * \param allocator to allocate memory for the copy of the name returned
     * \return a instance of smart pointer that would deallocate the buffer when out of scope.
     *  The OrtAllocator instances must be valid at the point of memory release.
     *)
    function GetOverridableInitializerNameAllocated(index: size_t;
      allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::SessionGetOverridableInitializerName

    (** \brief Returns a copy of the profiling file name.
     *
     * \param allocator to allocate memory for the copy of the ortstring returned
     * \return a instance of smart pointer that would deallocate the buffer when out of scope.
     *  The OrtAllocator instances must be valid at the point of memory release.
     *)
    function EndProfilingAllocated(allocator: TOrtAllocator): AllocatedStringPtr;  ///< Wraps OrtApi::SessionEndProfiling
    function GetProfilingStartTimeNs(): uint64_t;                                 ///< Wraps OrtApi::SessionGetProfilingStartTimeNs
    function GetModelMetadata() :TORTModelMetadata;                                   ///< Wraps OrtApi::SessionGetModelMetadata
    function GetInputTypeInfo(index:size_t) :TOrtTypeInfo;                   ///< Wraps OrtApi::SessionGetInputTypeInfo
    function GetOutputTypeInfo(index:size_t) :TOrtTypeInfo;                  ///< Wraps OrtApi::SessionGetOutputTypeInfo
    function GetOverridableInitializerTypeInfo( index:size_t) :TOrtTypeInfo;  ///< Wraps OrtApi::SessionGetOverridableInitializerTypeInfo

  end;


  { TTensorTypeAndShapeInfoHelper }

  TORTTensorTypeAndShapeInfoHelper = record helper for TORTTensorTypeAndShapeInfo                                                   ///< Create an empty TensorTypeAndShapeInfo object, must be assigned a valid one to be used
    function GetElementType():ONNXTensorElementDataType;  ///< Wraps OrtApi::GetTensorElementType
    function GetElementCount():size_t;                    ///< Wraps OrtApi::GetTensorShapeElementCount

    function GetDimensionsCount():size_t;                                           ///< Wraps OrtApi::GetDimensionsCount
    procedure GetDimensions( values:PInt64_t; values_count: size_t ) ;              ///< Wraps OrtApi::GetDimensions
    procedure GetSymbolicDimensions(const values:PPOrtChar; values_count:size_t) ; overload; ///< Wraps OrtApi::GetSymbolicDimensions
    function GetSymbolicDimensions(const values_count: size_t): TArray<ortstring>;
      overload;
    function GetShape():TArray<int64_t>;  ///< Uses GetDimensionsCount & GetDimensions to return a std::vector of the shape

  end;

  { TSequenceTypeInfoHelper }

  TORTSequenceTypeInfoHelper = record helper for TORTSequenceTypeInfo
    function GetSequenceElementType():TORTTypeInfo;  ///< Wraps OrtApi::GetSequenceElementType
  end;


  { TMapTypeInfoHelper }

  TORTMapTypeInfoHelper = record helper for TORTMapTypeInfo
    function GetMapKeyType() :ONNXTensorElementDataType;  ///< Wraps OrtApi::GetMapKeyType
    function GetMapValueType() :TORTTypeInfo;                 ///< Wraps OrtApi::GetMapValueType
  end;

  { TOrtTypeInfoHelper }

  TORTTypeInfoHelper = record helper for TORTTypeInfo
    function GetTensorTypeAndShapeInfo():TORTTensorTypeAndShapeInfo;  ///< Wraps OrtApi::CastTypeInfoToTensorInfo
    function GetSequenceTypeInfo():TORTSequenceTypeInfo;              ///< Wraps OrtApi::CastTypeInfoToSequenceTypeInfo
    function GetMapTypeInfo():TORTMapTypeInfo;                        ///< Wraps OrtApi::CastTypeInfoToMapTypeInfo
    function GetONNXType():ONNXType;
  end;


  { TValueHelper }

  TORTValueHelper = record helper for TORTValue
  type
    TOrtSparseValuesParam =record
      values_shape      : Pint64_t ;
      values_shape_len  : size_t ;
      data : record case boolean of
        false:(p_data:Pointer);
        true :(str:PPOrtChar);
      end;
    end;
    TShape = record
      shape:Pint64_t;
      shape_len:size_t;
    end;

    class function CreateTensor(const info:POrtMemoryInfo; p_data:pointer; p_data_byte_count:size_t ; const shape:Pint64_t; shape_len:size_t ; _type:ONNXTensorElementDataType ):TORTValue;   overload; static;
    class function CreateTensor(allocator:POrtAllocator; const shape:Pint64_t; shape_len:size_t ; _type:ONNXTensorElementDataType ):TORTValue;    overload;static;
    class function CreateMap(var keys :TORTValue;var values:TORTValue):TORTValue; static;      ///< Wraps OrtApi::CreateValue
    class function CreateSequence(var values:TArray<TORTValue>):TORTValue; static; ///< Wraps OrtApi::CreateValue
    function IsTensor():boolean;  ///< Returns true if Value is a tensor, false for other types like map/sequence/etc
    function HasValue():boolean;  /// < Return true if OrtValue contains data and returns false if the OrtValue is a None
    function GetCount():size_t;  // If a non tensor, returns 2 for map and N for sequence, where N is the number of elements
    function GetValue(index:longint; allocator:POrtAllocator) :TORTValue;
    function GetStringTensorDataLength() :size_t;
    procedure GetStringTensorContent( buffer:Pointer;  buffer_length:size_t; offsets:Psize_t; offsets_count:size_t) ;
    function GetTypeInfo():TOrtTypeInfo;
    function GetTensorTypeAndShapeInfo():TORTTensorTypeAndShapeInfo;
    function GetStringTensorElementLength(element_index: size_t): size_t;
    procedure GetStringTensorElement(buffer_length:size_t; element_index:size_t; buffer:pointer );
    procedure FillStringTensor(const s:PPOrtChar; s_len:size_t );
    procedure FillStringTensorElement(const s:POrtChar; index: size_t);
{$ifndef DISABLE_SPARSE_TENSORS}
    class function CreateSparseTensor(const info:POrtMemoryInfo;p_data: pointer ; const dense_shape:TShape; const values_shape:TShape; _type: ONNXTensorElementDataType):TORTValue;overload;static;
    procedure UseCooIndices(indices_data:Pint64_t;indices_num :size_t );
    procedure UseCsrIndices(inner_data:Pint64_t; inner_num:size_t; outer_data:Pint64_t; outer_num:size_t );
    procedure UseBlockSparseIndices(const indices_shape :TShape ;indices_data :Pint32_t);
    class function CreateSparseTensor(allocator:POrtAllocator; const dense_shape:TShape; _type: ONNXTensorElementDataType):TORTValue;    overload;static;
    procedure FillSparseTensorCoo(const data_mem_info:POrtMemoryInfo; const values_param:TOrtSparseValuesParam ;const indices_data:Pint64_t; indices_num:size_t );
    procedure FillSparseTensorCsr(const data_mem_info:POrtMemoryInfo;
                           const  values:TOrtSparseValuesParam;
                           const inner_indices_data:PInt64_t; inner_indices_num :size_t ;
                           const outer_indices_data:PInt64_t; outer_indices_num :size_t );
    procedure FillSparseTensorBlockSparse(const data_mem_info :POrtMemoryInfo;
                                   const values :TOrtSparseValuesParam;
                                   const  indices_shape :TShape;
                                   const  indices_data:Pint32_t);
    function GetSparseFormat():OrtSparseFormat;
    function GetSparseTensorValuesTypeAndShapeInfo(): TORTTensorTypeAndShapeInfo;
    function GetSparseTensorIndicesTypeShapeInfo(indices_format: OrtSparseIndicesFormat): TORTTensorTypeAndShapeInfo;
    function IsSparseTensor():boolean;
    class function CreateSparseTensor<T>(const info:POrtMemoryInfo;var p_data:T; const dense_shape:TShape; const values_shape:TShape ):TORTValue; overload; static;
    class function CreateSparseTensor<T>(allocator:POrtAllocator; const dense_shape:TShape):TORTValue;    overload;static;
    function GetSparseTensorIndicesData<PT>( indices_format:OrtSparseIndicesFormat;var num_indices: size_t):PT;  //T*
    function GetSparseTensorValues<PT>():PT; // T*
{$endif}
    class function CreateTensor<T>(const info:POrtMemoryInfo;var p_data: T ; p_data_element_count:size_t ; const  shape:Pint64_t; shape_len:size_t ):TORTValue;overload;static;
    class function CreateTensor<T>(allocator:POrtAllocator; const shape:Pint64_t;shape_len :size_t):TORTValue;      overload; static;
    class function CreateOpaque<T>(const domain:POrtChar; const type_name:POrtChar; const data_container:T):TORTValue;static;  ///< Wraps OrtApi::CreateOpaqueValue
    procedure GetOpaqueData<T>(const domain:POrtChar; const type_name:POrtChar;var _out:T);  ///< Wraps OrtApi::GetOpaqueValue
    function GetTensorMutableData<T>: Pointer;  /// T*< Wraps OrtApi::GetTensorMutableData
    function GetTensorData<T>: Pointer;  ///T* < Wraps OrtApi::GetTensorMutableData
    function GetTensorShape:TArray<int64_t>;
    function GettensorType:ONNXTensorElementDataType;
    function At<T>(const location:TArray<int64_t>):T;
  end;


  { TMemoryInfoHelper }

  TORTMemoryInfoHelper = record helper for TORTMemoryInfo
    class function CreateCpu(_type:OrtAllocatorType ; mem_type:OrtMemType ):TORTMemoryInfo;  static;
    class function Create(const name:POrtChar; _type:OrtAllocatorType; id:longint; mem_type:OrtMemType):TORTMemoryInfo; static;

    function GetAllocatorName():ortstring;
    function GetAllocatorType() :OrtAllocatorType;
    function GetDeviceId(): longint;
    function GetMemoryType() :OrtMemType;
    function GetDeviceType() :OrtMemoryInfoDeviceType;
    function Equal(const other:TORTMemoryInfo):boolean;
    //  equal == operator is implemented in TORTBase
  end;

  { TORTMemoryAllocation }

  TORTMemoryAllocation = record
    constructor Create(const allocator: POrtAllocator; const p: Pointer; const size: size_t);
    function get():pointer; { return p_; }
    function size():size_t; { return size_; }
    //{$ifdef fpc}
    //class operator Copy(constref Src:TORTMemoryAllocation; var dst:TORTMemoryAllocation);
    //{$else}
    //class operator Assign(var dst:TORTMemoryAllocation; const [ref] src:TORTMemoryAllocation);
    //{$endif}
    //class operator Initialize({$ifdef fpc}var{$else}out{$endif} dest:TORTMemoryAllocation);
    class operator Finalize(var dest:TORTMemoryAllocation);

  private
    allocator_ : POrtAllocator ;
    p_    : pointer ;
    size_ : size_t;
  end;

  { TAllocatorHelper }

  TORTAllocatorHelper = record helper for TORTAllocator
    class function Create(const session :TORTSession; const mem_info :TORTMemoryInfo):TORTAllocator;static;

    function Alloc(const size:size_t):pointer;
    // The return value will own the allocation
    function GetAllocation(size:size_t):TORTMemoryAllocation;
    procedure Free(p:pointer);
    function GetMemoryInfo():TORTMemoryInfo;    // result must have unowned content "implement a trick so it won't not be released if gone out of scope?"
    function GetInfo():POrtMemoryInfo;
  end;


  { TArenaCfgHelper }

  TORTArenaCfgHelper = record helper for TORTArenaCfg
     class function ArenaCfg( max_mem:size_t; arena_extend_strategy:longint; initial_chunk_size_bytes:longint; max_dead_bytes_per_chunk:longint):TORTArenaCfg;static;
  end;

  { TAllocatorWithDefaultOptions }

  TORTAllocatorWithDefaultOptions=record
    class operator Initialize({$ifdef fpc}var{$else}out{$endif} dest:TORTAllocatorWithDefaultOptions);
    class operator Finalize(var dest:TORTAllocatorWithDefaultOptions);
    class operator Implicit(const src:TORTAllocatorWithDefaultOptions):POrtAllocator;
    function Alloc(const size:size_t):Pointer;
    // The return value will own the allocation
    function GetAllocation(const size :size_t):TORTMemoryAllocation;
    procedure Free(const p:pointer);
    function GetInfo():POrtMemoryInfo;
    function GetMemoryInfo():TORTMemoryInfo;
    class operator Implicit(const v:TORTAllocatorWithDefaultOptions):TORTAllocator;
  public
    p_:POrtAllocator;
  end;

  {$ifdef fpc}
  { TCustomOpBase }
  TORTCustomOpBase<TOp,TKernel> =record
    type
      POp=^TOp;
      PKernel=^TKernel;
  public
    version : uint32_t;
    function CreateKernel(const this_:POrtCustomOp;const api:POrtApi;const info:POrtKernelInfo):pointer;stdcall;
    function GetName(const this_:POrtCustomOp):POrtChar;stdcall;
    function GetExecutionProviderType(const this_:POrtCustomOp):POrtChar;  overload; stdcall;
    function GetInputType(const this_:POrtCustomOp; index:size_t):ONNXTensorElementDataType;stdcall;
    function GetInputTypeCount(const this_:POrtCustomOp):size_t;stdcall;
    function GetOutputType(const this_:POrtCustomOp; index:size_t):ONNXTensorElementDataType;stdcall;
    function GetOutputTypeCount(const this_:POrtCustomOp):size_t;stdcall;
    procedure KernelCompute(op_kernel:pointer; context:POrtKernelContext);stdcall;
    procedure KernelDestroy(op_kernel:pointer);stdcall;
    function GetInputCharacteristic(const this_:POrtCustomOp; index:size_t):OrtCustomOpInputOutputCharacteristic;overload;stdcall;
    function GetOutputCharacteristic(const this_:POrtCustomOp; index:size_t):OrtCustomOpInputOutputCharacteristic;overload;stdcall;
    class operator Initialize({$ifdef fpc}var{$else}out{$endif} dest:TORTCustomOpBase<TOp,TKernel>);

    // Default implementation of GetExecutionProviderType that returns nullptr to default to the CPU provider
    function GetExecutionProviderType():POrtChar;overload;

    // Default implementations of GetInputCharacteristic() and GetOutputCharacteristic() below
    // (inputs and outputs are required by default)
    function GetInputCharacteristic(index: size_t):OrtCustomOpInputOutputCharacteristic;overload;

    function  GetOutputCharacteristic(index :size_t):OrtCustomOpInputOutputCharacteristic;overload;
  end;
  {$endif}
  { TCustomOpApi }

  TORTCustomOpApi = record
    constructor Create(const api:OrtApi);
   // T is only implemented for std::vector<float>, std::vector<int64_t>, float, int64_t, and ortstring
    function KernelInfoGetAttribute<T>(const info :POrtKernelInfo; const name: POrtChar): T;                                               inline;
    function GetTensorTypeAndShape(const value : POrtValue):POrtTensorTypeAndShapeInfo;                                                 inline;
    function GetTensorShapeElementCount(const info :POrtTensorTypeAndShapeInfo):size_t;                                                 inline;
    function GetTensorElementType(const info :POrtTensorTypeAndShapeInfo):ONNXTensorElementDataType;                                    inline;
    function GetDimensionsCount(const info :POrtTensorTypeAndShapeInfo):size_t;                                                         inline;
    procedure GetDimensions(const info: POrtTensorTypeAndShapeInfo; dim_values: Pint64_t; dim_values_length: size_t);                   inline;
    procedure SetDimensions(info :POrtTensorTypeAndShapeInfo; const dim_values:Pint64_t; dim_count:size_t);                             inline;
    function GetTensorMutableData<T>(value:POrtValue):Pointer;                                                                          inline;
    function GetTensorData<T>(const value:POrtValue):Pointer;                                                                           inline;
    function GetTensorMemoryInfo(const value:POrtValue):TOrtMemoryInfo;                                                                 inline;
    function GetTensorShape(const info: POrtTensorTypeAndShapeInfo):TArray<Int64_t>;                                                    inline;
    procedure ReleaseTensorTypeAndShapeInfo(input :POrtTensorTypeAndShapeInfo);                                                         inline;
    function KernelContext_GetInputCount(const context :POrtKernelContext):size_t;                                                      inline;
    function KernelContext_GetInput(const context: POrtKernelContext; index:size_t):POrtValue;                                          inline;
    function KernelContext_GetOutputCount(const context :POrtKernelContext):size_t;                                                     inline;
    function KernelContext_GetOutput(context :POrtKernelContext; index:size_t ;const dim_values:PInt64_t; dim_count:size_t): POrtValue; inline;
    function KernelContext_GetGPUComputeStream(const context:POrtKernelContext):Pointer;                                                inline;
    procedure ThrowOnError(status :POrtStatus);                                                                                         inline;
    function CreateOpAttr(const name :POrtChar; const data:Pointer; len :longint; _type :OrtOpAttrType ):POrtOpAttr;                       inline;
    procedure ReleaseOpAttr(op_attr: POrtOpAttr);                                                                                       inline;
    function CreateOp(const info:POrtKernelInfo;
                      const op_name:POrtChar;
                      const domain :POrtChar;
                      version:longint;
                      const type_constraint_names:PPOrtChar ;
                      const type_constraint_values:PONNXTensorElementDataType;
                      type_constraint_count :longint ;
                      const attr_values:PPOrtOpAttr;
                      attr_count:longint;
                      input_count:longint;
                      output_count:longint):POrtOp;                                                                                     inline;
    procedure InvokeOp(const context :POrtKernelContext;
                   const ort_op: POrtOp;
                   const input_values: PPOrtValue;
                   input_count: longint;
                   var output_values: PPOrtValue ;
                   output_count:longint);                                                                                               inline;
    procedure ReleaseOp(ort_op :POrtOp);                                                                                                inline;
    function CopyKernelInfo(const info :POrtKernelInfo):POrtKernelInfo;                                                                 inline;
    procedure ReleaseKernelInfo(info_copy:POrtKernelInfo );                                                                             inline;

  private
    api_ : OrtApi;

  end;

  TOrtTypeSingle     = record helper for single     const OrtTensorType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   ; end;
  TOrtTypeFloat16_t  = record helper for Float16_t  const OrtTensorType = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ; end;
  TOrtTypeBFloat16_t = record helper for BFloat16_t const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; end;
  TOrtTypedouble     = record helper for double     const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE  ; end;
  TOrtTypeint8_t     = record helper for int8_t     const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8    ; end;
  TOrtTypeint16_t    = record helper for int16_t    const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16   ; end;
  TOrtTypeint32_t    = record helper for int32_t    const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32   ; end;
  TOrtTypeint64_t    = record helper for int64_t    const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   ; end;
  TOrtTypeuint8_t    = record helper for uint8_t    const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8   ; end;
  TOrtTypeuint16_t   = record helper for uint16_t   const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16  ; end;
  TOrtTypeuint32_t   = record helper for uint32_t   const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32  ; end;
  TOrtTypeuint64_t   = record helper for uint64_t   const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64  ; end;
  TOrtTypeboolean    = record helper for boolean    const OrtTensortype = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL    ; end;
  var
    HouseKeeper: TMemHousekeeper;

    DefaultAllocator      :TORTAllocatorWithDefaultOptions;
    DefaultEnv            :TORTEnv;
    DefaultSessionOptions :TORTSessionOptions;
    DefaultRunOptions     :TORTRunOptions;

    function GetAvailableProviders():TArray<ortstring>;inline;
    procedure ThrowOnError(const ort:POrtApi ; status: POrtStatus);inline; overload;
    procedure ThrowOnError(status:POrtStatus);inline; overload;
    function GetApi:POrtApi;
    function AllocatorGetMemoryInfo(const Allocator:POrtAllocator):TORTMemoryInfo;
    function OrtTensorType(const typinf:PTypeInfo):ONNXTensorElementDataType;
    //function strcmp(_para1, _para2:POrtChar):longint;cdecl;external 'libc';
    procedure free(addr:pointer);cdecl;external 'libc';
    procedure cfree(addr:pointer);cdecl;external 'libc';
implementation


{$ifdef MEM_TABLE}
procedure THKHelp.print();
var i:integer;
begin
  for i:= 0 to High(keys) do
    if assigned(values[i]) then
      writeln(IntToStr(NativeUInt(keys[i])),' ==> ',IntToStr(values[i]^)) ;
  writeln('')
end;
{$endif}
function OrtTensorType(const typinf:PTypeInfo):ONNXTensorElementDataType;
begin
   if typinf=TypeInfo(single     ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   ;exit end;
   if typinf=TypeInfo(Float16_t  ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ;exit  end;
   if typinf=TypeInfo(BFloat16_t ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;exit  end;
   if typinf=TypeInfo(double     ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE  ;exit  end;
   if typinf=TypeInfo(int8_t     ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8    ;exit  end;
   if typinf=TypeInfo(int16_t    ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16   ;exit  end;
   if typinf=TypeInfo(int32_t    ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32   ;exit  end;
   if typinf=TypeInfo(int64_t    ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   ;exit  end;
   if typinf=TypeInfo(uint8_t    ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8   ;exit  end;
   if typinf=TypeInfo(uint16_t   ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16  ;exit  end;
   if typinf=TypeInfo(uint32_t   ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32  ;exit  end;
   if typinf=TypeInfo(uint64_t   ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64  ;exit  end;
   if typinf=TypeInfo(boolean    ) then begin result := ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL    ;exit  end;

end;
  { TOrtTensor }

function TORTTensor<T>.Getindex1( x: int64_t): T;
begin
  result:=FData[x]
end;

function TORTTensor<T>.Getindex3( x,y,z: int64_t): T;
begin
  result:=FData[idx(z,y,x)]
end;

function TORTTensor<T>.Getindex2( x,y: int64_t): T;
begin
  result:=FData[y*FShape[0]+x]
end;

function TORTTensor<T>.GetIndex4(x,y,z,w: int64_t): T;
begin
  result:=FData[w*FShape[2]*FShape[1]*FShape[0]+z*FShape[1]*FShape[0]+y*FShape[0]+x]
end;

procedure TORTTensor<T>.Setindex1( x: int64_t; AValue: T);
begin
  FData[x]:=AValue
end;

procedure TORTTensor<T>.Setindex3( x, y, z: int64_t; AValue: T);
begin
  FData[z*FShape[1]*FShape[0]+y*FShape[0]+x]:=AValue
end;

procedure TORTTensor<T>.Setindex2(x,y: int64_t; AValue: T);
begin
  FData[y*FShape[0]+x]:=AValue
end;

procedure TORTTensor<T>.SetIndex4(x,y,z,w: int64_t; AValue: T);
begin
  FData[w*FShape[2]*FShape[1]*FShape[0]+z*FShape[1]*FShape[0]+y*FShape[0]+x]:=AValue
end;

procedure TORTTensor<T>.SetShape( AValue: TArray<int64_t>);
var i:size_t;
begin
  //if FShape=AValue then Exit;
  if not Assigned(AValue) then exit;
  FShape:=AValue;
  FSize:=FShape[0];
  for i:=1 to High(FShape) do
    FSize:=FSize*FShape[i];
end;

function TORTTensor<T>.ElementCount: size_t;
begin
  result:=FSize
end;

function TORTTensor<T>.ByteCount: size_t;
begin
  result:=FSize*SizeOf(T)
end;

function TORTTensor<T>.DimensionCount: size_t;
begin
  result:=Length(FShape)
end;

function TORTTensor<T>.Idx(const x, y, z: Int64_t): int64_t;
begin
  result:=z*FShape[1]*FShape[0]+y*FShape[0]+x
end;

constructor TORTTensor<T>.Create(const AShape: TArray<int64_t>);
var aData:TArray<T>;
begin
  SetShape(aShape);
  setLength(aData,FSize);
  FData:=aData
end;

{$ifdef fpc}

function TORTTensor<T>.ToString(const Separator: ortstring): ortstring;
var
  z,y,x: Int64;s,s2:ortstring;
begin
  result:='';
  case Length(FShape) of
    3: begin
         for z:=0 to FShape[2]-1 do begin
           s2:='';
           for y:=0 to FShape[1]-1 do begin
             s:='';
             for x:=0 to FShape[0]-1 do begin
               s:=s+Separator+Format('%0.2n',[Index3[z,y,x] ])
             end;
             delete(s,1,length(LineEnding+Separator));
             s2:=s2+LineEnding+Separator+'[ '+s+' ]';
           end;
           delete(s2,1,length(LineEnding+Separator));
           result:=result+LineEnding+Separator+'[ '+s2+' ]';
         end;
         delete(result,1,length(Separator));
         result:='[ '+result+' ]';
       end;
    2: begin
         for y:=0 to FShape[1]-1 do begin
           s:='';
           for x:=0 to FShape[0]-1 do begin
             s:=s+Separator+Format('%0.2n',[Index2[y,x] ])
           end;
           delete(s,1,length(Separator));
           result:=result+LineEnding+Separator+'[ '+s+' ]';
         end;
         delete(result,1,length(LineEnding+Separator));
         result:='[ '+result+' ]';
       end;
    1: begin
         for x:=0 to FShape[0]-1 do begin
           result:=result+Separator+Format('%0.2n',[FData[x] ])
         end;
         delete(result,1,length(Separator));
         result:='[ '+result+' ]';
       end
  end;
end;
{$endif}

function TORTTensor<T>.ToValue: TORTValue;
var revShape:TArray<int64_t>;
begin
  revShape:=TTools.reverse<int64_t>(FShape);
  result:=TORTValue.CreateTensor<T>(DefaultAllocator.GetInfo(), FData[0],ElementCount,@revShape[0],DimensionCount);
end;

class function TORTTensor<T>.FromValue(const val: TORTValue;
  const copyData: boolean): TORTTensor<T>;
var Info:TOrtTypeInfo ;
    Meta:TORTTensorTypeAndShapeInfo;
    _data:PT;
begin
  // the data pointer is owned by the OrtValue and
 //  will be freed when the OrtValue is released

  try
    Info:=Val.GetTypeInfo();
    if Info.GetONNXType<>ONNX_TYPE_TENSOR then
      OrtException.CreateFmt('Value is not a Tensor %d',[Ord(Info.GetONNXType)]);
    Meta:=Info.GetTensorTypeAndShapeInfo;
    if OrtTensorType(TypeInfo(T))<>Meta.GetElementType then
      OrtException.CreateFmt('Tensor must be of Type [%s]',[ORTTensorTypes[Meta.GetElementType]]);
    ThrowOnError(GetApi().GetTensorMutableData(val.p_,@_data));
    if copyData then begin
      result:=TORTTensor<T>.Create(TTools.reverse<int64_t>(Val.GetTensorShape()));
      move(_data^,result.Fdata[0],SizeOf(T)*result.FSize);
    end else begin
      result.Shape:=TTools.reverse<int64_t>(Val.GetTensorShape());
      result.FData:=TArray<T>(_data)
    end;

  except
  end;
end;

class operator TORTTensor<T>.Implicit(const aValue:TORTValue): TORTTensor<T>;
begin
  result:=TORTTensor<T>.FromValue(aValue)
end;

class operator TORTTensor<T>.Implicit(const aValue: TORTTensor<T>): TORTValue;
begin
  result:=aValue.ToValue
end;

{$ifndef NO_HASHMAP}
{ TNameValueListHelper }

function TORTNameValue<TKey,TValue>.GetKeys(idx: size_t): TKey;
begin
  result:=inherited Keys.ToArray[idx];
end;

function TORTNameValue<TKey,TValue>.GetValues(idx: Size_t): TValue;
begin
  result:=inherited Values.ToArray[idx];
end;

{$endif}

class function TTools.BinSearch<T>(const Arr: array of T; const Val: T; R: integer): integer;
var
  L, I: Integer;
  CompareRes: Integer;isFound:boolean;
begin
  isFound := false;
  result:=-1;

  // Use binary search.
  L := 0;
  R := R - 1;
  while (L<=R) do
  begin
    I := L + (R - L) shr 1;
    CompareRes := cmp<T>(Val, Arr[I]);
    if (CompareRes>0) then
      L := I+1
    else begin
      R := I-1;
      if (CompareRes=0) then begin
         isFound := true;
//         if (Duplicates<>dupAccept) then
            L := I; // forces end of while loop
      end;
    end;
  end;
  if isFound then result := L else result:=-L-1;
end;


class function TTools.IfThen<T>(const cond: boolean; const whenTrue, whenFalse: T): T;
begin
  if cond then result:=whenTrue else result:=whenFalse
end;

class function TTools.max<T>(const a, b: T): T;
begin
  {$ifdef fpc}
  if a>=b then result:=a else result:=b
  {$else}
  result:=ifthen<T>(TComparer<T>.Default().Compare(a,b)>0,a,b)
  {$endif}
end;

class function TTools.min<T>(const a, b: T): T;
begin
  {$ifdef fpc}
  if a<=b then result:=a else result:=b
  {$else}
  result:=ifthen<T>(TComparer<T>.Default().Compare(a,b)<0,a,b)
  {$endif}
end;

class procedure TTools.QuickSort<T>(var Arr: array of T; L, R : Longint; const Descending:boolean;const Compare:TComparefunc<T>);
var I,J ,neg :integer;
    P, Q :T;
begin
 //if not Assigned(Compare) then Compare:=@{$ifdef fpc}specialize{$endif}_Compare<T>;
 neg:=ifthen<integer>(descending,-1,1);
 repeat
   I := L;
   J := R;
   P := Arr[ (L + R) div 2 ];
   repeat
     while neg*Compare(P, Arr[i]) > 0 do
       I := I + 1;
     while neg*Compare(P, Arr[J]) < 0 do
       J := J - 1;
     If I <= J then
     begin
       Q := Arr[I];
       Arr[I] := Arr[J];
       Arr[J] := Q;
       I := I + 1;
       J := J - 1;
     end;
   until I > J;
   if J - L < R - I then
   begin
     if L < J then
       QuickSort<T>(Arr, L, J, Descending, Compare);
     L := I;
   end
   else
   begin
     if I < R then
       QuickSort<T>(Arr, I, R, Descending, Compare);
     R := J;
   end;
 until L >= R;
end;

class function TTools.cmp<T>(const a, b: T): integer;
begin
  {$ifdef fpc}
  result:=1;
  if a<b then result:=-1
  else if a=b then result:=0
  {$else}
  result:=TComparer<T>.Default.Compare(a,b)
  {$endif}
end;

class function TTools.reverse<T>(const a: TArray<T>): TArray<T>;
var i,j:size_t;v:T;
begin
  result:=copy(a);
  i:=0; j:=high(a);
  while i<j do begin
    v:=result[i];
    result[i]:=result[j];
    result[j]:=v;
    inc(i);dec(j);
  end
end;

{$ifdef NO_HASHMAP}
{ TOrderedKeyValueList }

class operator TOrderedKeyValueList<TK,TV>.Initialize({$ifdef fpc}outvar{$else}out{$endif} v:TOrderedKeyValueList<TK,TV>);
begin
  v.FindKey:=nil// change to TTools.BinSearch<TK> for fast sorted keys
end;

class operator TOrderedKeyValueList<TK,TV>.Finalize(var dst:TOrderedKeyValueList<TK,TV>);
begin
  {$ifdef MEM_TABLE}writeln('***** [',GetTypeName(TypeInfo(TOrderedKeyValueList<TK,TV>)) ,'] Deleted ******') {$endif}
end;

function TOrderedKeyValueList<TK,TV>.Count: integer;
begin
  result:=Length(Keys)
end;

procedure TOrderedKeyValueList<TK, TV>.RemoveIndex(const index: integer);
begin
  if (index>=0) and (index<Count) then begin
    Delete(Keys,index,1);
    Delete(Values,index,1);
  end;
end;

procedure TOrderedKeyValueList<TK, TV>.Remove(const key: TK);
begin
  RemoveIndex(IndexOfKey(key))
end;

function TOrderedKeyValueList<TK,TV>.GetValues( key: TK): TV;
var i:integer;
begin
  i:= IndexOfKey(Key);
  if i>=0 then
    result:=Values[i]
end;

procedure TOrderedKeyValueList<TK,TV>.SetValues( key: TK;  AValue: TV);
var i:integer;
begin
  i:=IndexOfKey(Key);
  if i<0 then
    begin
      i:=not i;
      Insert(key, Keys,i);
      Insert(AValue,Values,i)
    end
  else
    //begin
      Values[i]:=AValue
    //end;
end;

class function TOrderedKeyValueList<TK, TV>.Create: TOrderedKeyValueList<TK, TV>;
begin
  // dummy constructor
end;

{$POINTERMATH ON}


function TOrderedKeyValueList<TK, TV>.TryGetValue(const Key: TK; out Value: TV): boolean;
var i,j:integer;
  P,k:PPointer;

begin
  p:=PPointer(keys);
  k:=@key;
  i:=IndexOfKey(Key);
  {$ifdef MEM_TABLE}
  write('locating.. ',NativeUInt(k^),' in ');
  for j := 0 to High(keys) do
    write(NativeUInt(p[j]),' , ');
  writeln('');
  writeln('found in ... ',i);
  {$endif}
  result:= i>-1;
  if result then
    Value:=Values[i]
end;

function TOrderedKeyValueList<TK,TV>.ContainsKey(const key: TK): boolean;
begin
  result:=IndexOfKey(key)>=0;
end;

function TOrderedKeyValueList<TK,TV>.IndexOfKey(const Key: TK): integer;
begin
  if assigned(FindKey) then begin
    result:=Findkey(Keys,key,Length(Keys));
    exit
  end;
  for result:=0 to high(Keys) do
    if TTools.cmp<TK>(Keys[result],key)=0 then exit ;
  result:=-Length(Keys)-1
end;

procedure TOrderedKeyValueList<TK, TV>.Add(const Key: TK; const AValue: TV);
var i:integer;
begin
  i:=IndexOfKey(Key);
  if i<0 then
    begin
      i:=-(i+1);
      Insert(key, Keys,i);
      Insert(AValue,Values,i)
    end
//  else
//      ValueList[i]:=AValue ;
end;

function TOrderedKeyValueList<TK, TV>.AddOrSetValue(const Key: TK; const AValue: TV): integer;
var i:integer;
begin
  i:=IndexOfKey(Key);
  if i<0 then
    begin
      i:=-(i+1);
      Insert(key, Keys,i);
      Insert(AValue,Values,i)
    end
  else
    //begin
      Values[i]:=AValue ;
  result:=i
    //end;
end;

{$endif}

{$ifndef NO_SMARTPTR}
{ TSmartPtr }

procedure TSmartPtr<T>.DecRef;
begin
  if assigned(RefCount) then
    {$ifdef fpc}
    if InterLockedDecrement(RefCount^)=0 then
    {$else}
    if TInterlocked.Decrement(RefCount^)=0 then
    {$endif}
    begin
      Dispose(RefCount);
      FreeMem(Instance)
    end;
end;

class operator TSmartPtr<T>.Initialize({$ifdef fpc}outvar{$else}out{$endif} dst: TSmartPtr<T>);
begin
  dst.RefCount := nil;
end;

class operator TSmartPtr<T>.Finalize(var dst: TSmartPtr<T>);
begin
  dst.DecRef();
end;

{$ifdef fpc}
class operator TSmartPtr<T>.AddRef(var src: TSmartPtr<T>);
begin
  if assigned(src.RefCount) then
    InterLockedIncrement(src.RefCount^);
end;
{$endif}

{$ifdef fpc}
class operator TSmartPtr<T>.Copy(constref src: TSmartPtr<T>; var dst: TSmartPtr<T>);
{$else}
class operator TSmartPtr<T>.Assign(var dst: TSmartPtr<T>;const [ref] src: TSmartPtr<T>);
{$endif}
begin
  if assigned(dst.RefCount) then
    dst.DecRef();
  if assigned(src.RefCount) then
  {$ifdef fpc}
    InterLockedIncrement(src.RefCount^);
  {$else}
  TInterlocked.Increment(src.RefCount^);
  {$endif}
  dst.RefCount := src.RefCount;
  dst.Instance := src.Instance;

end;

class operator TSmartPtr<T>.Implicit(const src: PT): TSmartPtr<T>;
begin
  Result.Assign(src);
end;

//class operator TSmartPtr<T>.Implicit(const src: TSmartPtr<T>): T;
//begin
//  Result:=Instance
//end;

procedure TSmartPtr<T>.Assign(const val: PT);
begin
  if RefCount <> nil then
    DecRef();

  New(RefCount);
  RefCount^ := 0;

  {$ifdef fpc}
  InterLockedIncrement(RefCount^);
  {$else}
  TInterlocked.Increment(RefCount^);
  {$endif}
  Instance := Val;
end;

//**********************************************************************************


{ TSmartPtr }

procedure TSmartPtr<T,P>.DecRef;
begin
  if RefCount <> nil then
    {$ifdef fpc}
    if InterLockedDecrement(RefCount^)=0 then
    {$else}
    if TInterLocked.Decrement(RefCount^)=0 then
    {$endif}
    begin
      Dispose(RefCount);
//      if Assigned(DisposerFunc) then
        DisposerFunc.call(Pointer(Instance));

      //else FreeMem(Instance);
    end;
end;

constructor TSmartPtr<T,P>.Create(const val:PT;const aDisposer: TORTAllocatedFree);
begin
  DisposerFunc:=aDisposer;
  assign(val)
end;

class operator TSmartPtr<T,P>.Initialize({$ifdef fpc}outvar{$else}out{$endif} dst: TSmartPtr<T,P>);
begin
  dst.RefCount := nil;
end;

class operator TSmartPtr<T,P>.Finalize(var dst: TSmartPtr<T,P>);
begin
  dst.DecRef();
end;
{$ifdef fpc}
class operator TSmartPtr<T,P>.AddRef(var src: TSmartPtr<T,P>);
begin
  if assigned(src.RefCount) then
    InterLockedIncrement(src.RefCount^);
  //if assigned(dst.RefCount) then

end;
{$endif}

{$ifdef fpc}
class operator TSmartPtr<T,P>.Copy(constref src: TSmartPtr<T,P>; var dst: TSmartPtr<T,P>);
{$else}
class operator TSmartPtr<T,P>.Assign(var dst: TSmartPtr<T,P>; const [ref] src: TSmartPtr<T,P>);
{$endif}
begin
  if dst.RefCount <> nil then
    dst.DecRef();
  if src.RefCount <> nil then
    {$ifdef fpc}
    InterLockedIncrement(src.RefCount^);
    {$else}
    TInterLocked.Increment(src.RefCount^);
    {$endif}
  dst.RefCount := src.RefCount;
  dst.Instance := src.Instance;
  //if assigned(dst.RefCount) then

end;

class operator TSmartPtr<T,P>.Implicit(const src: PT): TSmartPtr<T,P>;
begin
  Result.Assign(src);
  //if assigned(dst.RefCount) then
end;

//class operator TSmartPtr<T, P>.Implicit(const src: TSmartPtr<T, P>): T;
//begin
//  result:=Instance
//end;

procedure TSmartPtr<T,P>.Assign(const val: PT);
begin
  if RefCount <> nil then
    DecRef();

  New(RefCount);
  RefCount^ := 0;

  {$ifdef fpc}
  InterLockedIncrement(RefCount^);
  {$else}
  TInterLocked.Increment(RefCount^);
  {$endif}
  Instance := Val;
end;
{$endif}


function AllocatorGetMemoryInfo(const Allocator: POrtAllocator): TORTMemoryInfo;
begin
  ThrowOnError(GetApi().AllocatorGetInfo(Allocator, @Result.p_))
end;

procedure ThrowOnError(const ort:POrtApi ; status: POrtStatus);inline; overload;
var error_message:ortstring;error_code:OrtErrorCode;
begin
  if assigned(status) then begin
     error_message := ortstring(ort.GetErrorMessage(status));
     error_code := ort.GetErrorCode(status);
    ort.ReleaseStatus(status);
{$ifdef ORT_NO_EXCEPTION}
  Writeln(error_message, error_code);abort
{$else}
  raise OrtException.create(error_message, error_code)
{$endif}
  end;
end;

function GetApi:POrtApi;
begin
  result:=Api
end;

{ TORTAllocatedFree }

constructor TORTAllocatedFree.Create(const allocator: POrtAllocator);
begin
  allocator_:=allocator
end;

procedure TORTAllocatedFree.call(var ptr: pointer);
begin
  if Assigned(ptr) then allocator_.Free(allocator_,ptr);
end;

procedure ThrowOnError(status:POrtStatus);inline; overload;
begin
  ThrowOnError(Api, status);
end;

function GetAvailableProviders: TArray<ortstring>;
var len:longint;providers:PPOrtChar;i:integer;
begin
  Api.GetAvailableProviders(@Providers,@len);
  setLength(result,len);
  for i:=0 to High(result) do begin
    result[i]:=ortstring(Providers^);
    inc(Providers)
  end;
end;
//
//procedure OrtRelease(ptr: POrtAllocator);
//begin
//  GetApi.ReleaseAllocator(ptr);
//end;
//
//procedure OrtRelease(ptr: POrtMemoryInfo);
//begin
//  GetApi.ReleaseMemoryInfo(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtCustomOpDomain);
//begin
//  GetApi.ReleaseCustomOpDomain(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtEnv);
//begin
//  GetApi.ReleaseEnv(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtRunOptions);
//begin
//  GetApi.ReleaseRunOptions(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtSession);
//begin
//  GetApi.ReleaseSession(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtSessionOptions);
//begin
//  GetApi.ReleaseSessionOptions(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtTensorTypeAndShapeInfo);
//begin
//  GetApi.ReleaseTensorTypeAndShapeInfo(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtSequenceTypeInfo);
//begin
//  GetApi.ReleaseSequenceTypeInfo(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtMapTypeInfo);
//begin
//  GetApi.ReleaseMapTypeInfo(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtTypeInfo);
//begin
//  GetApi.ReleaseTypeInfo(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtValue);
//begin
//  GetApi.ReleaseValue(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtModelMetadata);
//begin
//  GetApi.ReleaseModelMetadata(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtThreadingOptions);
//begin
//  GetApi.ReleaseThreadingOptions(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtIoBinding);
//begin
//  GetApi.ReleaseIoBinding(ptr)
//end;
//
//procedure OrtRelease(ptr: POrtArenaCfg);
//begin
//  GetApi.ReleaseArenaCfg(ptr)
//end;

{ TBase }

procedure TORTBase<T>.NewRef;
var RefCount:PLongInt;
begin

 New(RefCount);
 {$IFDEF MEM_DEBUG}
 writeln('New @',IntToHex(UIntPtr(p_)),'[',PTypeInfo(TypeInfo(T)).Name,'] @Count [', intToHex(UIntPtr(RefCount)),']');
 {$ENDIF}
 RefCount^ := 0;
 HouseKeeper.AddOrSetValue(p_,RefCount);
 {$ifdef fpc}
 InterLockedIncrement(RefCount^);
 {$else}
 TInterLocked.Increment(RefCount^);
 {$endif}
 {$ifdef MEM_TABLE}writeln(GetTypeName(TypeInfo(T)),sLineBreak,'===== new =====',NativeUInt(p_));housekeeper.print;{$endif}
end;

procedure TORTBase<T>.Assign(const val: Pointer);
var RefCount:PLongInt;
begin
  if not HouseKeeper.TryGetValue(p_,RefCount) then RefCount:=nil;
  if RefCount <> nil then
    DecRef();
  p_ := Val;
  NewRef();

end;

procedure TORTBase<T>.DecRef;
var RefCount:PLongInt;
begin
  if not HouseKeeper.TryGetValue(p_,RefCount) then RefCount:=nil;
  {$IFDEF MEM_DEBUG}
  writeln('disposing @',IntToHex(UIntPtr(p_)),'[',PTypeInfo(TypeInfo(T)).Name,'] @count [',IntToHex(UIntPtr(RefCount)),']') ;
  {$ENDIF}
  {$ifdef MEM_TABLE}writeln(GetTypeName(TypeInfo(T)),sLineBreak,'===== Dec =====',NativeUInt(p_));{$endif}
  if assigned(RefCount) then begin
    {$IFDEF MEM_DEBUG}
    writeln('  -----> Decrimenting count [',RefCount^,'] to [',RefCount^-1,']') ;
    {$ENDIF}
    {$ifdef fpc}
    if InterLockedDecrement(RefCount^)=0 then
    {$else}
    if TInterLocked.Decrement(RefCount^)=0 then
    {$endif}
    begin
      Dispose(RefCount);
      HouseKeeper.Remove(p_);
      OrtRelease;
      {$IFDEF MEM_DEBUG}writeln('  ---> disposed') ;{$ENDIF}
    end;
    {$ifdef MEM_TABLE}housekeeper.print;writeln('');{$endif}
  end ;
  {$IFDEF MEM_DEBUG}Writeln('');  {$ENDIF}
end;

//function TORTBase<T>.RefCount: PLongInt;
//begin
//  result:=ReferenceCount(p_);
//end;

function TORTBase<T>.release: PT;
var p:PT;
begin
  p:=p_;
  p_:=nil;
  result:=p
end;

procedure TORTBase<T>.OrtRelease();
begin
  (* Filler Procedure *)
  if p_=nil then exit;
  if TypeInfo(T)=TypeInfo(OrtAllocator)                 then begin Api.ReleaseAllocator(p_)             ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtMemoryInfo)                then begin Api.ReleaseMemoryInfo(p_)            ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtCustomOpDomain)            then begin Api.ReleaseCustomOpDomain(p_)        ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtEnv)                       then begin Api.ReleaseEnv(p_)                   ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtRunOptions)                then begin Api.ReleaseRunOptions(p_)            ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtSession)                   then begin Api.ReleaseSession(p_)               ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtSessionOptions)            then begin Api.ReleaseSessionOptions(p_)        ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtTensorTypeAndShapeInfo)    then begin Api.ReleaseTensorTypeAndShapeInfo(p_);  exit  end;
  if TypeInfo(T)=TypeInfo(OrtSequenceTypeInfo)          then begin Api.ReleaseSequenceTypeInfo(p_)      ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtMapTypeInfo)               then begin Api.ReleaseMapTypeInfo(p_)           ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtTypeInfo)                  then begin Api.ReleaseTypeInfo(p_)              ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtValue)                     then
  begin
    Api.ReleaseValue(p_)
        ;  exit
  end;
  if TypeInfo(T)=TypeInfo(OrtModelMetadata)             then begin Api.ReleaseModelMetadata(p_)         ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtIoBinding)                 then begin Api.ReleaseIoBinding(p_)             ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtArenaCfg)                  then begin Api.ReleaseArenaCfg(p_)              ;  exit  end;
  if TypeInfo(T)=TypeInfo(OrtThreadingOptions)          then begin Api.ReleaseThreadingOptions(p_)      ;  exit  end;
  //if TypeInfo(T)=TypeInfo(OrtCheckpointState)           then begin Api.ReleaseCheckpointState(p_)       ;  exit  end;
  //if TypeInfo(T)=TypeInfo(OrtTrainingSession)           then begin Api.ReleaseTrainingSession(p_)       ;  exit  end;
end;

constructor TORTBase<T>.Create(const v: PT);
begin
  p_:=v
  //Assign(v);
end;

class operator TORTBase<T>.Initialize({$ifdef fpc}outvar{$else}out{$endif} v: TORTBase<T>);
var RefCount:PLongInt;
begin
  v.p_:=nil;
  if GetApi=nil then exit;
  if TypeInfo(T)=TypeInfo(OrtSessionOptions) then begin
      ThrowOnError(GetApi().CreateSessionOptions(@v.p_));
      if Assigned(DefaultSessionOptions.p_) then v.NewRef;
      exit
  end;
  if TypeInfo(T)=TypeInfo(OrtRunOptions) then begin
      ThrowOnError(GetApi().CreateRunOptions(@v.p_));
      if Assigned(DefaultRunOptions.p_) then v.NewRef;
      exit
  end;
end;

class operator TORTBase<T>.Finalize(var v: TORTBase<T>);
begin
  if TypeInfo(T)=TypeInfo(OrtTensorTypeAndShapeInfo) then
    begin v.release; exit end;    //unowned casted types
  if TypeInfo(T)=TypeInfo(OrtSequenceTypeInfo) then
      begin v.release; exit end;
  if TypeInfo(T)=TypeInfo(OrtMapTypeInfo) then
      begin v.release; exit end;
  //if TypeInfo(T)=TypeInfo(OrtMemoryInfo) then
  //    begin v.release;exit end;

  //if TypeInfo(T)=TypeInfo(OrtTypeInfo) then
  //    begin v.release;exit end;

  //if TypeInfo(T)=TypeInfo(OrtEnv) then
  //  if HouseKeeper.ContainsKey(v.p_) then
  //    writeln('Finalize, OrtEnv, RefCount: ', HouseKeeper[v.p_]^);
  v.DecRef ;

end;

{$ifdef fpc}
class operator TORTBase<T>.Copy(constref src: TORTBase<T>; var dst: TORTBase<T>);
{$else}
class operator TORTBase<T>.Assign(var dst: TORTBase<T>; const [ref] src: TORTBase<T>);
{$endif}
var dRefCount,sRefCount:PLongInt;
begin
  if not HouseKeeper.TryGetValue(dst.p_,dRefCount) then dRefCount:=nil;
  if not HouseKeeper.TryGetValue(src.p_,sRefCount) then sRefCount:=nil;
  if assigned(dRefCount) then
    dst.DecRef();
  {$IFDEF MEM_DEBUG}
  writeln('Passing @',IntToHex(UIntPtr(src.p_)),'[',PTypeInfo(TypeInfo(T)).Name,'] @sCount [', intToHex(UIntPtr(sRefCount)),']');
  writeln('To        @',IntToHex(UIntPtr(dst.p_)),'[',PTypeInfo(TypeInfo(T)).Name,'] @dCount [',intToHex(UIntPtr(dRefCount)),']');
  {$ENDIF}
  {$ifdef MEM_TABLE}writeln(GetTypeName(TypeInfo(T)),sLineBreak,'===== asn =====',NativeUInt(src.p_),' , ',NativeUInt(dst.p_));{$endif}
  if assigned(sRefCount) then begin
    {$IFDEF MEM_DEBUG}
    writeLn('  -----> incrementing sCount[',sRefCount^,'] to [',sRefCount^+1,']');
    {$ENDIF}
    {$ifdef fpc}
    InterLockedIncrement(sRefCount^);
    {$else}
    TInterLocked.Increment(sRefCount^);
    {$endif}
//    {$ifdef MEM_TABLE}writeln(GetTypeName(TypeInfo(T)),sLineBreak,'===== asn =====');housekeeper.print;{$endif}
  end;
  if Assigned(dst.p_)then begin
    housekeeper.AddOrSetValue(dst.p_, sRefCount);
    {$ifdef MEM_TABLE}housekeeper.print;{$endif}
  end;
  dst.p_ := src.p_;
  {$IFDEF MEM_DEBUG}Writeln('');  {$ENDIF}
 end;

{$ifdef fpc}
class operator TORTBase<T>.AddRef(var src: TORTBase<T>);
var RefCount:PLongInt;
begin
  if not HouseKeeper.TryGetValue(src.p_,RefCount) then RefCount:=nil;
  {$IFDEF MEM_DEBUG}
  writeln('Adding Ref @',IntToHex(UIntPtr(src.p_)),'[',PTypeInfo(TypeInfo(T)).Name,'] @sCount [', intToHex(UIntPtr(RefCount)),']');
  {$ENDIF}
  {$ifdef MEM_TABLE}writeln(GetTypeName(TypeInfo(T)),sLineBreak,'===== new =====');{$endif}
  if assigned(RefCount) then begin
    {$IFDEF MEM_DEBUG}Writeln('   -----> Incrementing count [',RefCount^,'] to [',RefCount^+1,']');  {$ENDIF}
    InterLockedIncrement(RefCount^);
    {$IFDEF MEM_DEBUG}
    writeln('   Added --->  @',IntToHex(UIntPtr(src.p_)),'[',PTypeInfo(TypeInfo(T)).Name,'] sCount [', RefCount^,']');
    {$ENDIF}
    {$ifdef MEM_TABLE}housekeeper.print;{$endif}
  end
  //if TypeInfo(T)=TypeInfo(OrtEnv) then
  //  writeln('AddRef, OrtEnv, RefCount: ', HouseKeeper[src.p_]^);
  {$IFDEF MEM_DEBUG}Writeln('');  {$ENDIF}
end;
{$endif}
//class operator TORTBase<T>.Implicit(const v: TORTBase<T>): PT;
//begin
//  result:=v.p_;
//end;

{.$PackRecords 1}
class operator TORTBase<T>.Implicit(const v: PT): TORTBase<T>;
begin
  result.Assign(v) ;
  //if TypeInfo(T)=TypeInfo(OrtEnv) then
  //  writeln('Implicit, OrtEnv, RefCount: ', HouseKeeper[result.p_]^);

  if not Assigned(v) then
    raise OrtException.Create(format('Allocation failure: %s',[TTypeInfo(TypeInfo(T)^).name]),ORT_FAIL)
end;

{.$PackRecords C}

class operator TORTBase<T>.Equal(const a, b: TORTBase<T>): boolean;
var comp:longint;
begin
  if TypeInfo(T)=TypeInfo(OrtMemoryInfo) then begin
      result:=false;
      ThrowOnError(GetApi().CompareMemoryInfo(a.p_, b.p_, @comp));
      result:=comp=0
  end else
    result:=CompareMem(@a,@b,SizeOf(a));
end;

{ TORTIoBindingHelper }

class function TORTIoBindingHelper.IoBinding(var session: TORTSession):TORTIoBinding;
begin
  ThrowOnError(GetApi().CreateIoBinding(session.p_, @result.p_));
  result.NewRef;
end;

procedure TORTIoBindingHelper.BindInput(const name: POrtChar; const value: TORTValue);
begin
  ThrowOnError(GetApi().BindInput(p_, name, value.p_));
end;

procedure TORTIoBindingHelper.BindOutput(const name: POrtChar; const value: TORTValue);
begin
  ThrowOnError(GetApi().BindOutput(p_, name, value.p_));
end;

procedure TORTIoBindingHelper.BindOutput(const name: POrtChar; const mem_info: TORTMemoryInfo);
begin
  ThrowOnError(GetApi().BindOutputToDevice(p_, name, mem_info.p_));
end;

function TORTIoBindingHelper.GetOutputNames: TArray<ortstring>;
var allocator:TORTAllocatorWithDefaultOptions;
begin
  result := GetOutputNamesHelper(allocator.p_);
end;

function TORTIoBindingHelper.GetOutputNames(var alloc: TORTAllocator): TArray<ortstring>;
begin
  result := GetOutputNamesHelper(alloc.p_);
end;

function TORTIoBindingHelper.GetOutputValues: TArray<TORTValue>;
var allocator: TORTAllocatorWithDefaultOptions;
begin
  result:= GetOutputValuesHelper(allocator.p_);
end;

function TORTIoBindingHelper.GetOutputValues(var alloc: TORTAllocator): TArray<TORTValue>;
begin
  result:= GetOutputValuesHelper(alloc.p_);
end;

procedure TORTIoBindingHelper.ClearBoundInputs;
begin
  GetApi().ClearBoundInputs(p_);
end;

procedure TORTIoBindingHelper.ClearBoundOutputs;
begin
  GetApi().ClearBoundOutputs(p_);
end;

procedure TORTIoBindingHelper.SynchronizeInputs;
begin
  ThrowOnError(GetApi().SynchronizeBoundInputs(p_));
end;

procedure TORTIoBindingHelper.SynchronizeOutputs;
begin
  ThrowOnError(GetApi().SynchronizeBoundOutputs(p_));
end;

function TORTIoBindingHelper.GetOutputNamesHelper(allocator: POrtAllocator): TArray<ortstring>;
var count,i,sz:size_t;buffer:POrtChar;lengths:Psize_t;
  {$ifndef NO_SMARTPTR}
  free_fn:TORTAllocatedFree;
  buffer_g:TSmartPtr<OrtChar,TORTAllocatedFree>;lengths_g:TSmartPtr<size_t,TORTAllocatedFree>;
  {$endif}
begin
  {$ifndef NO_SMARTPTR}free_fn.allocator_:=allocator; {$endif}

  ThrowOnError(GetApi().GetBoundOutputNames(p_, allocator, @buffer, @lengths, @count));
  {$ifndef NO_SMARTPTR}
  buffer_g.DisposerFunc:=free_fn;
  buffer_g:=@buffer;
  lengths_g.DisposerFunc:=free_fn;
  lengths_g:=@lengths;
  {$endif}
  if count = 0 then exit;
  setLength(result,count);
  for i := 0 to high(result)do begin
    sz := lengths^;
    result[i]:=copy(ortstring(buffer),0,sz);
    inc(lengths);
    inc(buffer, sz);
  end;
end;

function TORTIoBindingHelper.GetOutputValuesHelper(allocator: POrtAllocator): TArray<TORTValue>;
var output_count,i:size_t;output_buffer:PPOrtValue;
  {$ifndef NO_SMARTPTR}
  free_fn:TORTAllocatedFree;
  buffer_g:TSmartPtr<POrtValue,TORTAllocatedFree>;
  {$endif}
begin
  // Lambda to release the buffer when no longer needed and
  // make sure that we destroy all instances on exception
  {$ifndef NO_SMARTPTR}
  free_fn.allocator_:=allocator;
  buffer_g.DisposerFunc:=free_fn;
  {$endif}
  ThrowOnError(GetApi().GetBoundOutputValues(p_, allocator, @output_buffer, @output_count));
  if output_count=0 then exit;
  {$ifndef NO_SMARTPTR}
  buffer_g:=@output_buffer;
  {$endif}
  setLength(result,output_count);
  for i := 0 to high(result) do begin
    result[i]:=pointer(output_buffer^); // OrtValue will be released on Finalization operator implemented by TBase<>
    inc(output_buffer)
  end;
  allocator.free(allocator,output_buffer)

end;

{ TORTArenaCfgHelper }

class function TORTArenaCfgHelper.ArenaCfg(max_mem: size_t;
  arena_extend_strategy: longint; initial_chunk_size_bytes: longint;
  max_dead_bytes_per_chunk: longint):TORTArenaCfg;
begin
  ThrowOnError(GetApi().CreateArenaCfg(max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk, @result.p_));
  result.NewRef;
end;

{ TORTAllocatorHelper }

class function TORTAllocatorHelper.Create(const session: TORTSession; const mem_info: TORTMemoryInfo):TORTAllocator;
begin
  ThrowOnError(GetApi().CreateAllocator(session.p_, mem_info.p_, @result.p_));
  result.NewRef();
end;

function TORTAllocatorHelper.Alloc(const size: size_t): pointer;
begin
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, @result));
end;

function TORTAllocatorHelper.GetAllocation(size: size_t): TORTMemoryAllocation;
var _out:pointer;
begin
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, @_out));
  result:=TORTMemoryAllocation.Create(p_, _out, size);
end;

procedure TORTAllocatorHelper.Free(p: pointer);
begin
  ThrowOnError(GetApi().AllocatorFree(p_, p));
end;

function TORTAllocatorHelper.GetMemoryInfo: TORTMemoryInfo;
//var _out:POrtMemoryInfo;
begin
  ThrowOnError(GetApi().AllocatorGetInfo(p_, @result));
  //result:= _out;
end;

function TORTAllocatorHelper.GetInfo: POrtMemoryInfo;
begin
  ThrowOnError(GetApi().AllocatorGetInfo(p_,@result))
end;

{ TORTMemoryInfoHelper }

class function TORTMemoryInfoHelper.CreateCpu(_type: OrtAllocatorType;
  mem_type: OrtMemType):TORTMemoryInfo;
begin
  ThrowOnError(GetApi().CreateCpuMemoryInfo(_type, mem_type, @result.p_));
end;

class function TORTMemoryInfoHelper.Create(const name: POrtChar;
  _type: OrtAllocatorType; id: longint; mem_type: OrtMemType):TORTMemoryInfo;
begin
  ThrowOnError(GetApi().CreateMemoryInfo(name, _type, id, mem_type, @result.p_));
end;

function TORTMemoryInfoHelper.GetAllocatorName: ortstring;
var name:POrtChar;
begin
  ThrowOnError(GetApi().MemoryInfoGetName(p_, @name));
  result:=ortstring(name)
end;

function TORTMemoryInfoHelper.GetAllocatorType: OrtAllocatorType;
begin
  ThrowOnError(GetApi().MemoryInfoGetType(p_, @result));
end;

function TORTMemoryInfoHelper.GetDeviceId: longint;
begin
  ThrowOnError(GetApi().MemoryInfoGetId(p_, @result));
end;

function TORTMemoryInfoHelper.GetMemoryType: OrtMemType;
begin
  ThrowOnError(GetApi().MemoryInfoGetMemType(p_, @result));
end;

function TORTMemoryInfoHelper.GetDeviceType():OrtMemoryInfoDeviceType;
begin
  GetApi().MemoryInfoGetDeviceType(p_, @result);
end;

function TORTMemoryInfoHelper.Equal(const other: TORTMemoryInfo): boolean;
var comp_result:longint;
begin
  ThrowOnError(GetApi().CompareMemoryInfo( self.p_, other.p_, @comp_result));
  result:= comp_result = 0;
end;

{ TORTValueHelper }

class function TORTValueHelper.CreateTensor(const info: POrtMemoryInfo;
  p_data: pointer; p_data_byte_count: size_t; const shape: Pint64_t;
  shape_len: size_t; _type: ONNXTensorElementDataType):TORTValue;
begin
  ThrowOnError(GetApi().CreateTensorWithDataAsOrtValue(info, p_data, p_data_byte_count, shape, shape_len, _type, @result.p_));
  result.NewRef();
end;

class function TORTValueHelper.CreateTensor(allocator: POrtAllocator;
  const shape: Pint64_t; shape_len: size_t; _type: ONNXTensorElementDataType):TORTValue;
begin
  ThrowOnError(GetApi().CreateTensorAsOrtValue(allocator, shape, shape_len, _type, @result.p_));
  result.NewRef();
end;

class function TORTValueHelper.CreateMap(var keys: TORTValue; var values: TORTValue):TORTValue;
var inputs:array [0..1] of POrtValue;
begin
  inputs[0]:=keys.p_;inputs[1]:=values.p_;
  ThrowOnError(GetApi().CreateValue(@inputs[0], 2, ONNX_TYPE_MAP, @result.p_));
  result.NewRef;
end;

class function TORTValueHelper.CreateSequence(var values: TArray<TORTValue>):TORTValue;
begin
  ThrowOnError(GetApi().CreateValue(@values[0], length(values), ONNX_TYPE_SEQUENCE, @result.p_));
  result.NewRef();
end;

function TORTValueHelper.IsTensor: boolean;
begin
  ThrowOnError(GetApi().IsTensor(p_, @result));
end;

function TORTValueHelper.HasValue: boolean;
begin
  ThrowOnError(GetApi().HasValue(p_, @result));
end;

// for non tensor types  Only ( will return 2 case of a map ,  Count of sequences in case of sequence)
function TORTValueHelper.GetCount: size_t;
begin
  ThrowOnError(GetApi().GetValueCount(p_, @result));
end;

// for non tensor types  Only ( index is 0 or 1 in case of a map , 0 to N in case of sequence)
function TORTValueHelper.GetValue(index: longint; allocator: POrtAllocator): TORTValue;
begin
  ThrowOnError(GetApi().GetValue(p_, index, allocator, @result.p_));
  result.NewRef;
end;

function TORTValueHelper.GetStringTensorDataLength: size_t;
begin
  ThrowOnError(GetApi().GetStringTensorDataLength(p_, @result));
end;

procedure TORTValueHelper.GetStringTensorContent(buffer: Pointer;
  buffer_length: size_t; offsets: Psize_t; offsets_count: size_t);
begin
  ThrowOnError(GetApi().GetStringTensorContent(p_, buffer, buffer_length, offsets, offsets_count));
end;

function TORTValueHelper.GetTypeInfo: TOrtTypeInfo;
begin
  ThrowOnError(GetApi().GetTypeInfo(p_, @result.p_));
end;

function TORTValueHelper.GetTensorTypeAndShapeInfo: TORTTensorTypeAndShapeInfo;
begin
  ThrowOnError(GetApi().GetTensorTypeAndShape(p_, @result.p_));
end;

function TORTValueHelper.GetStringTensorElementLength(element_index: size_t): size_t;
begin
  ThrowOnError(GetApi().GetStringTensorElementLength(p_, element_index, @result));
end;

procedure TORTValueHelper.GetStringTensorElement(buffer_length: size_t;
  element_index: size_t; buffer: pointer);
begin
  ThrowOnError(GetApi().GetStringTensorElement(p_, buffer_length, element_index, buffer));
end;

procedure TORTValueHelper.FillStringTensor(const s: PPOrtChar; s_len: size_t);
begin
  ThrowOnError(GetApi().FillStringTensor(p_, s, s_len));
end;

procedure TORTValueHelper.FillStringTensorElement(const s: POrtChar; index: size_t);
begin
  ThrowOnError(GetApi().FillStringTensorElement(p_, s, index));
end;
{$ifndef DISABLE_SPARSE_TENSORS}
class function TORTValueHelper.CreateSparseTensor(const info: POrtMemoryInfo;
  p_data: pointer; const dense_shape: TShape; const values_shape: TShape;
  _type: ONNXTensorElementDataType):TORTValue;
begin
  ThrowOnError(GetApi().CreateSparseTensorWithValuesAsOrtValue(info, p_data, dense_shape.shape, dense_shape.shape_len,
                                                               values_shape.shape, values_shape.shape_len, _type, @result.p_));
  result.NewRef();
end;

procedure TORTValueHelper.UseCooIndices(indices_data: Pint64_t; indices_num: size_t);
begin
  ThrowOnError(GetApi().UseCooIndices(p_, indices_data, indices_num));
end;

procedure TORTValueHelper.UseCsrIndices(inner_data: Pint64_t; inner_num: size_t;
  outer_data: Pint64_t; outer_num: size_t);
begin
  ThrowOnError(GetApi().UseCsrIndices(p_, inner_data, inner_num, outer_data, outer_num));
end;

procedure TORTValueHelper.UseBlockSparseIndices(const indices_shape: TShape;
  indices_data: Pint32_t);
begin
  ThrowOnError(GetApi().UseBlockSparseIndices(p_, indices_shape.shape, indices_shape.shape_len, indices_data));
end;

class function TORTValueHelper.CreateSparseTensor(allocator: POrtAllocator;
  const dense_shape: TShape; _type: ONNXTensorElementDataType):TORTValue;
begin
  ThrowOnError(GetApi().CreateSparseTensorAsOrtValue(allocator, dense_shape.shape, dense_shape.shape_len, _type, result.p_));
  result.NewRef();
end;

procedure TORTValueHelper.FillSparseTensorCoo(const data_mem_info: POrtMemoryInfo;
  const values_param: TOrtSparseValuesParam; const indices_data: Pint64_t;
  indices_num: size_t);
begin
  ThrowOnError(GetApi().FillSparseTensorCoo(p_, data_mem_info, values_param.values_shape,
                                            values_param.values_shape_len, values_param.data.p_data,
                                            indices_data, indices_num));
end;

procedure TORTValueHelper.FillSparseTensorCsr(const data_mem_info: POrtMemoryInfo;
  const values: TOrtSparseValuesParam; const inner_indices_data: PInt64_t;
  inner_indices_num: size_t; const outer_indices_data: PInt64_t;
  outer_indices_num: size_t);
begin
  ThrowOnError(GetApi().FillSparseTensorCsr(p_, data_mem_info, values.values_shape, values.values_shape_len, values.data.p_data,
                                            inner_indices_data, inner_indices_num,
                                            outer_indices_data, outer_indices_num));
end;

procedure TORTValueHelper.FillSparseTensorBlockSparse(
  const data_mem_info: POrtMemoryInfo; const values: TOrtSparseValuesParam;
  const indices_shape: TShape; const indices_data: Pint32_t);
begin
  ThrowOnError(GetApi().FillSparseTensorBlockSparse(p_, data_mem_info, values.values_shape, values.values_shape_len, values.data.p_data,
                                                    indices_shape.shape, indices_shape.shape_len,
                                                    indices_data));
end;

function TORTValueHelper.GetSparseFormat: OrtSparseFormat;
begin
  ThrowOnError(GetApi().GetSparseTensorFormat(p_, @result));
end;

function TORTValueHelper.GetSparseTensorValuesTypeAndShapeInfo: TORTTensorTypeAndShapeInfo;
begin
  ThrowOnError(GetApi().GetSparseTensorValuesTypeAndShape(p_, @result));
end;

function TORTValueHelper.GetSparseTensorIndicesTypeShapeInfo(
  indices_format: OrtSparseIndicesFormat): TORTTensorTypeAndShapeInfo;
begin
  ThrowOnError(GetApi().GetSparseTensorIndicesTypeShape(p_, indices_format, @result));
end;

function TORTValueHelper.IsSparseTensor: boolean;
begin
  ThrowOnError(GetApi().IsSparseTensor(p_, @result))
end;

class function TORTValueHelper.CreateSparseTensor<T>(const info: POrtMemoryInfo;
  var p_data: T; const dense_shape: TShape; const values_shape: TShape): TORTValue;
begin
  result:= TORTValue.CreateSparseTensor(info, @p_data, dense_shape, values_shape, OrtTensorType(TypeInfo(T)));
end;

class function TORTValueHelper.CreateSparseTensor<T>(allocator: POrtAllocator; const dense_shape: TShape): TORTValue;
begin
  result:=TORTValue.CreateSparseTensor(allocator, dense_shape, OrtTensorType(TypeInfo(T)));
end;

function TORTValueHelper.GetSparseTensorIndicesData<PT>(
  indices_format: OrtSparseIndicesFormat; var num_indices: size_t): PT;
begin
  ThrowOnError(GetApi().GetSparseTensorIndices(p_, indices_format, @num_indices, @result));
end;

function TORTValueHelper.GetSparseTensorValues<PT>: PT;
begin
  ThrowOnError(GetApi().GetSparseTensorValues(p_, @result))
end;
{$endif}
class function TORTValueHelper.CreateTensor<T>(const info: POrtMemoryInfo;
  var p_data: T; p_data_element_count: size_t; const shape: Pint64_t;
  shape_len: size_t): TORTValue;
begin     // no need for NewRef here the constructor will create one ?!
  result:=TORTValue.CreateTensor(info, @p_data, p_data_element_count * sizeof(PT), shape, shape_len, OrtTensorType(TypeInfo(T)));
end;

class function TORTValueHelper.CreateTensor<T>(allocator: POrtAllocator;
  const shape: Pint64_t; shape_len: size_t): TORTValue;
begin     // no need for NewRef here the constructor will create one ?!
  result:= TORTValue.CreateTensor(allocator, shape, shape_len, OrtTensorType(TypeInfo(T)));
end;

class function TORTValueHelper.CreateOpaque<T>(const domain: POrtChar;
  const type_name: POrtChar; const data_container: T): TORTValue;
begin
  ThrowOnError(GetApi().CreateOpaqueValue(domain, type_name, @data_container, sizeof(T), @result));
  result.NewRef
end;

procedure TORTValueHelper.GetOpaqueData<T>(const domain: POrtChar;
  const type_name: POrtChar; var _out: T);
begin
  ThrowOnError(GetApi().GetOpaqueValue(domain, type_name, p_, @_out, sizeof(T)));
end;

function TORTValueHelper.GetTensorMutableData<T>: Pointer;
begin
  ThrowOnError(GetApi().GetTensorMutableData(p_,@result));
end;

function TORTValueHelper.GetTensorData<T>: Pointer;
begin
  ThrowOnError(GetApi().GetTensorMutableData(p_, @result));
end;

function TORTValueHelper.GetTensorShape: TArray<int64_t>;
begin
  result:=GetTensorTypeAndShapeInfo().GetShape();
end;

function TORTValueHelper.GetTensorType: ONNXTensorElementDataType;
begin
 result:=GetTensorTypeAndShapeInfo().GetElementType();
end;

function TORTValueHelper.At<T>(const location: TArray<int64_t>): T;
type  PT=^T;
var _out:PT;
begin
  // must not be a string
  ThrowOnError(GetApi().TensorAt(p_, @location[0], length(location), @_out));
  result:=_out^
end;

{ TOrtTypeInfoHelper }

function TOrtTypeInfoHelper.GetTensorTypeAndShapeInfo: TORTTensorTypeAndShapeInfo;
begin
  ThrowOnError(GetApi().CastTypeInfoToTensorInfo(p_, @result));
end;

function TOrtTypeInfoHelper.GetSequenceTypeInfo: TORTSequenceTypeInfo;
begin
  ThrowOnError(GetApi().CastTypeInfoToSequenceTypeInfo(p_, @result));
end;

function TOrtTypeInfoHelper.GetMapTypeInfo: TORTMapTypeInfo;
begin
  ThrowOnError(GetApi().CastTypeInfoToMapTypeInfo(p_, @result));
end;

function TOrtTypeInfoHelper.GetONNXType: ONNXType;
begin
  ThrowOnError(GetApi().GetOnnxTypeFromTypeInfo(p_, @result));
end;

{ TORTMapTypeInfoHelper }

function TORTMapTypeInfoHelper.GetMapKeyType: ONNXTensorElementDataType;
begin
  ThrowOnError(GetApi().GetMapKeyType(p_, @result));
end;

function TORTMapTypeInfoHelper.GetMapValueType: TORTTypeInfo;
begin
  ThrowOnError(GetApi().GetMapValueType(p_, @result));
end;

{ TORTSequenceTypeInfoHelper }

function TORTSequenceTypeInfoHelper.GetSequenceElementType: TORTTypeInfo;
begin
  ThrowOnError(GetApi().GetSequenceElementType(p_, @result));
end;

{ TORTTensorTypeAndShapeInfoHelper }

function TORTTensorTypeAndShapeInfoHelper.GetElementType: ONNXTensorElementDataType;
begin
  ThrowOnError(GetApi().GetTensorElementType(p_, @result));
end;

function TORTTensorTypeAndShapeInfoHelper.GetElementCount: size_t;
begin
  ThrowOnError(GetApi().GetTensorShapeElementCount(p_, @result));
end;

function TORTTensorTypeAndShapeInfoHelper.GetDimensionsCount: size_t;
begin
  ThrowOnError(GetApi().GetDimensionsCount(p_, @result));
end;

procedure TORTTensorTypeAndShapeInfoHelper.GetDimensions(values: PInt64_t;values_count: size_t);
begin
  ThrowOnError(GetApi().GetDimensions(p_, values, values_count));
end;
// below propably needs an allocation before calling? what is the count size needed??
procedure TORTTensorTypeAndShapeInfoHelper.GetSymbolicDimensions(
  const values: PPOrtChar; values_count: size_t);
begin
  ThrowOnError(GetApi().GetSymbolicDimensions(p_, values, values_count));
end;

function TORTTensorTypeAndShapeInfoHelper.GetSymbolicDimensions(const values_count:size_t): TArray<ortstring>;
var i:size_t;strs:PPOrtChar;
begin
  GetSymbolicDimensions(strs,values_count);
  setLength(result,values_count);
  for i:=0 to High(result) do begin
    result[i]:=ortstring(strs^);
    inc(strs)
  end;
end;

function TORTTensorTypeAndShapeInfoHelper.GetShape: TArray<int64_t>;
begin
  setLength(result,GetDimensionsCount());
  GetDimensions(@result[0], length(result));
end;

{ TSessionHelper }

class function TORTSessionHelper.Create(const env: TORTEnv;
  const model_path: PORTCHAR_T; const options: TORTSessionOptions):TORTSession;
begin
  ThrowOnError(GetApi().CreateSession(env.p_, model_path, options.p_, @result.p_));
  result.NewRef;
end;

class function TORTSessionHelper.Create(const env: TORTEnv;
  const model_path: PORTCHAR_T; const options: TORTSessionOptions;
  prepacked_weights_container: POrtPrepackedWeightsContainer):TORTSession;
begin
  ThrowOnError(GetApi().CreateSessionWithPrepackedWeightsContainer(env.p_, model_path, options.p_, prepacked_weights_container, @result.p_));
  result.NewRef;
end;

class function TORTSessionHelper.Create(const env: TORTEnv; const model_data: Pointer;
  model_data_length: size_t; const options: TORTSessionOptions):TORTSession;
begin
  ThrowOnError(GetApi().CreateSessionFromArray(env.p_, model_data, model_data_length, options.p_, @result.p_));
  result.NewRef;
end;

class function TORTSessionHelper.Create(const env: TORTEnv; const model_data: Pointer;
  model_data_length: size_t; const options: TORTSessionOptions;
  prepacked_weights_container: POrtPrepackedWeightsContainer):TORTSession;
begin
  ThrowOnError(GetApi().CreateSessionFromArrayWithPrepackedWeightsContainer(env.p_, model_data, model_data_length, options.p_,
                                                                            prepacked_weights_container, @result.p_));
  result.NewRef;
end;

class function TORTSessionHelper.Create(const model_path: TFileName):TORTSession;
var _path:widestring;
begin
  _path:=model_path;
  ThrowOnError(GetApi().CreateSession(DefaultEnv.p_, PORTCHAR_T(_path), DefaultSessionOptions.p_, @result.p_));
  result.NewRef;
end;

function Join(const arr:array of int64_t; const Delimiter:ortstring=', '):ortstring;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+IntToStr(arr[i]);
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

function TORTSessionHelper.run(const run_options: TORTRunOptions; const Inputs: TORTNameValueList): TORTNameValueList;
begin
  Run(run_options,Inputs,DefaultAllocator)
end;

function TORTSessionHelper.run(const run_options: TORTRunOptions; const Inputs: TORTNameValueList; Allocator: TOrtAllocator): TORTNameValueList;
var
  InputNames:TArray<ortstring>;
  InputValues:TArray<TORTValue>;
  OutputNames:TArray<ortstring>;
  OutputValues:TArray<TORTValue>;
  i:size_t;
begin
  setLength(InputNames,GetInputCount());
  setLength(InputValues,length(InputNames));
  setLength(OutputNames,GetOutputCount());
  setLength(OutputValues,length(OutputNames));
  for i:=0 to high(InputNames) do begin
    InputNames[i]:=ortstring(GetInputNameAllocated(i,Allocator){$ifndef NO_SMARTPTR}.Instance{$endif});
    InputValues[i]:=Inputs[ortstring(InputNames[i])];
  end;
  for i:=0 to High(OutputNames) do begin
    OutputNames[i]:=ortstring(GetOutputNameAllocated(i,Allocator){$ifndef NO_SMARTPTR}.Instance{$endif});
  end;
  Run(run_options,@InputNames[0],@InputValues[0],length(InputNames),
                  @OutputNames[0],@OutputValues[0],Length(OutputNames));
  {$ifndef NO_HASHMAP}result:=TNameValueList.Create;{$endif}
  for i:=0 to High(OutputNames) do begin
    OutputValues[i].NewRef;  // make sure a reference is created to enforce freeing out of scope housekeeping
    result.AddOrSetValue(OutputNames[i],OutputValues[i]);
  end;

end;

function TORTSessionHelper.Run(const run_options: TORTRunOptions;
  const input_names: PPOrtChar; const input_values: PORTValue; input_count: size_t;
  const output_names: PPOrtChar; output_names_count: size_t): TArray<TORTValue>;
var i:size_t;
begin
  setLength(result,output_names_count);
  Run(run_options, input_names, input_values, input_count,
                  output_names, @result[0], output_names_count);
  for i := 0 to output_names_count-1 do
    result[i].NewRef;// create reference for housekeeping

end;

function TORTSessionHelper.Run(const Inputs: TORTNameValueList): TORTNameValueList;
begin
  result:=Run(DefaultRunOptions, Inputs, DefaultAllocator)
end;

procedure TORTSessionHelper.Run(const run_options: TORTRunOptions;
  const input_names: PPOrtChar; const input_values: PORTValue; input_count: size_t;
  const output_names: PPOrtChar; output_values: PORTValue; output_count: size_t);
begin
  {$if SizeOf(TORTValue)<>SizeOf(POrtValue)}
    {$error Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely}
  {$endif}
  //assert(sizeof(TORTValue) = sizeof(POrtValue), 'Value is really just an array of OrtValue* in memory, so we can reinterpret_cast safely');
  //auto ort_input_values = reinterpret_cast<const OrtValue**>(const_cast<Value*>(input_values));
  //auto ort_output_values = reinterpret_cast<OrtValue**>(output_values);
  ThrowOnError(GetApi().Run(p_, run_options.p_, input_names, @input_values^, input_count, output_names, output_count, @output_values^));
end;

procedure TORTSessionHelper.Run(const run_options: TORTRunOptions;  const io_binding: TORTIoBinding);
begin
  ThrowOnError(GetApi().RunWithBinding(p_, run_options.p_, io_binding.p_));
end;

function TORTSessionHelper.GetInputCount: size_t;
begin
  ThrowOnError(GetApi().SessionGetInputCount(p_, @result));
end;

function TORTSessionHelper.GetOutputCount: size_t;
begin
  ThrowOnError(GetApi().SessionGetOutputCount(p_, @result));
end;

function TORTSessionHelper.GetOverridableInitializerCount: size_t;
begin
  ThrowOnError(GetApi().SessionGetOverridableInitializerCount(p_, @result));
end;

function TORTSessionHelper.GetInputNameAllocated(index: size_t;
  allocator: TOrtAllocator): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().SessionGetInputName(p_, index, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(allocator,_out);{$endif}
end;

function TORTSessionHelper.GetOutputNameAllocated(index: size_t;
  allocator: TOrtAllocator): AllocatedStringPtr;
var  _out:POrtChar;
begin
  ThrowOnError(GetApi().SessionGetOutputName(p_, index, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(allocator,_out);{$endif}
end;

function TORTSessionHelper.GetOverridableInitializerNameAllocated(index: size_t;
  allocator: TOrtAllocator): AllocatedStringPtr;
var  _out:POrtChar;
begin
  ThrowOnError(GetApi().SessionGetOverridableInitializerName(p_, index, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(allocator,_out);{$endif}
end;

function TORTSessionHelper.EndProfilingAllocated(allocator: TOrtAllocator
  ): AllocatedStringPtr;
var  _out:POrtChar;
begin
  ThrowOnError(GetApi().SessionEndProfiling(p_, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(allocator,_out);{$endif}
end;

function TORTSessionHelper.GetProfilingStartTimeNs: uint64_t;
begin
  ThrowOnError(GetApi().SessionGetProfilingStartTimeNs(p_, @result));
end;

function TORTSessionHelper.GetModelMetadata: TORTModelMetadata;
begin
  ThrowOnError(GetApi().SessionGetModelMetadata(p_, @result.p_));
  result.NewRef;
end;

function TORTSessionHelper.GetInputTypeInfo(index: size_t): TOrtTypeInfo;
begin
  ThrowOnError(GetApi().SessionGetInputTypeInfo(p_, index, @result.p_)) ;
end;

function TORTSessionHelper.GetOutputTypeInfo(index: size_t): TOrtTypeInfo;
begin
  ThrowOnError(GetApi().SessionGetOutputTypeInfo(p_, index, @result.p_));
end;

function TORTSessionHelper.GetOverridableInitializerTypeInfo(index: size_t ): TOrtTypeInfo;
begin
  ThrowOnError(GetApi().SessionGetOverridableInitializerTypeInfo(p_, index, @result.p_));
end;

{ TORTModelMetadataHelper }

function TORTModelMetadataHelper.GetProducerNameAllocated(allocator: TOrtAllocator
  ): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().ModelMetadataGetProducerName(p_, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(Allocator,_out);{$endif}
end;

function TORTModelMetadataHelper.GetGraphNameAllocated(allocator: TOrtAllocator
  ): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().ModelMetadataGetGraphName(p_, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(Allocator,_out);{$endif}
end;

function TORTModelMetadataHelper.GetDomainAllocated(allocator: TOrtAllocator
  ): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().ModelMetadataGetDomain(p_, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(Allocator,_out);{$endif}
end;

function TORTModelMetadataHelper.GetDescriptionAllocated(allocator: TOrtAllocator
  ): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().ModelMetadataGetDescription(p_, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(Allocator,_out);{$endif}
end;

function TORTModelMetadataHelper.GetGraphDescriptionAllocated(allocator: TOrtAllocator
  ): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().ModelMetadataGetGraphDescription(p_, allocator.p_, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(Allocator,_out);{$endif}
end;

function TORTModelMetadataHelper.GetCustomMetadataMapKeysAllocated(
  allocator: TOrtAllocator): TArray<AllocatedStringPtr>;
var _out:PPOrtChar; num_keys,i:int64_t;
begin
  ThrowOnError(GetApi().ModelMetadataGetCustomMetadataMapKeys(p_, allocator.p_, @_out, @num_keys));
  if num_keys <= 0 then
    exit;

  setLength(result,num_keys);
  // array of pointers will be freed
  // reserve may throw
  for i := 0 to num_keys-1 do begin
      {$ifndef NO_SMARTPTR}result[i].DisposerFunc:=TORTAllocatedFree.Create(allocator.p_);{$endif}
      result[i]:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out^);{$else}ortstring(_out^) {$endif} ;
      inc(_out)
  end;
  {$ifdef NO_SMARTPTR}
  for i := 0 to num_keys-1 do
    allocator.Free(Allocator,_out[i]);
  {$endif}
end;

function TORTModelMetadataHelper.LookupCustomMetadataMapAllocated(
  const key: POrtChar; allocator: TOrtAllocator): AllocatedStringPtr;
var _out:POrtChar;
begin
  ThrowOnError(GetApi().ModelMetadataLookupCustomMetadataMap(p_, allocator.p_, key, @_out));
  {$ifndef NO_SMARTPTR}result.DisposerFunc:=TORTAllocatedFree.create(allocator.p_);{$endif}
  result:={$ifndef NO_SMARTPTR}AllocatedStringPtr.PT(_out);{$else}ortstring(_out) {$endif}
  {$ifdef NO_SMARTPTR}allocator.Free(Allocator,_out);{$endif}
end;

function TORTModelMetadataHelper.GetVersion: int64_t;
begin
  ThrowOnError(GetApi().ModelMetadataGetVersion(p_, @result));
end;

{ TORTSessionOptionsHelper }

function TORTSessionOptionsHelper.Clone: TORTSessionOptions;
begin
  ThrowOnError(GetApi().CloneSessionOptions(p_, @result.p_));
end;

function TORTSessionOptionsHelper.SetIntraOpNumThreads(
  intra_op_num_threads: longint): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetIntraOpNumThreads(p_, intra_op_num_threads));
  result:=Self
end;

function TORTSessionOptionsHelper.SetInterOpNumThreads(
  inter_op_num_threads: longint): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetInterOpNumThreads(p_, inter_op_num_threads));
  result:=Self
end;

function TORTSessionOptionsHelper.SetGraphOptimizationLevel(
  graph_optimization_level: GraphOptimizationLevel): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetSessionGraphOptimizationLevel(p_, graph_optimization_level));
  Result:=Self
end;

function TORTSessionOptionsHelper.EnableCpuMemArena: TORTSessionOptions;
begin
  ThrowOnError(GetApi().EnableCpuMemArena(p_));
  Result:=Self
end;

function TORTSessionOptionsHelper.DisableCpuMemArena: TORTSessionOptions;
begin
  ThrowOnError(GetApi().DisableCpuMemArena(p_));
  Result:=Self
end;

function TORTSessionOptionsHelper.SetOptimizedModelFilePath(
  const optimized_model_filepath: PORTCHAR_T): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetOptimizedModelFilePath(p_, optimized_model_filepath));
  Result:=Self
end;

function TORTSessionOptionsHelper.EnableProfiling(
  const profile_file_prefix: PORTCHAR_T): TORTSessionOptions;
begin
  ThrowOnError(GetApi().EnableProfiling(p_, profile_file_prefix));
  Result:=Self
end;

function TORTSessionOptionsHelper.DisableProfiling: TORTSessionOptions;
begin
  ThrowOnError(GetApi().DisableProfiling(p_));
  result:=Self
end;

function TORTSessionOptionsHelper.EnableOrtCustomOps: TORTSessionOptions;
begin
  ThrowOnError(GetApi().EnableOrtCustomOps(p_));
  result:=Self
end;

function TORTSessionOptionsHelper.EnableMemPattern: TORTSessionOptions;
begin
  ThrowOnError(GetApi().EnableMemPattern(p_));
  result:=Self
end;

function TORTSessionOptionsHelper.DisableMemPattern: TORTSessionOptions;
begin
  ThrowOnError(GetApi().DisableMemPattern(p_));
  result:=Self
end;

function TORTSessionOptionsHelper.SetExecutionMode(execution_mode: ExecutionMode): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetSessionExecutionMode(p_, execution_mode));
  result:=Self
end;

function TORTSessionOptionsHelper.SetLogId(const logid: POrtChar): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetSessionLogId(p_, logid));
  result:=Self
end;

function TORTSessionOptionsHelper.SetLogSeverityLevel(level: Longint
  ): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SetSessionLogSeverityLevel(p_, level));
  result:=Self
end;

function TORTSessionOptionsHelper.Add(custom_op_domain: POrtCustomOpDomain
  ): TORTSessionOptions;
begin
  ThrowOnError(GetApi().AddCustomOpDomain(p_, custom_op_domain));
  result:=Self
end;

function TORTSessionOptionsHelper.DisablePerSessionThreads: TORTSessionOptions;
begin
  ThrowOnError(GetApi().DisablePerSessionThreads(p_));
  result:=Self
end;

function TORTSessionOptionsHelper.AddConfigEntry(const config_key: POrtChar;
  const config_value: POrtChar): TORTSessionOptions;
begin
  ThrowOnError(GetApi().AddSessionConfigEntry(p_, config_key, config_value));
  result:=Self
end;

function TORTSessionOptionsHelper.AddInitializer(const name: POrtChar;
  const ort_val: POrtValue): TORTSessionOptions;
begin
  ThrowOnError(GetApi().AddInitializer(p_, name, ort_val));
  result:=Self
end;

function TORTSessionOptionsHelper.AddExternalInitializers(
  const names: TArray<ortstring>; const ort_values: array of TORTValue): TORTSessionOptions;
var
  inputs_num,i:size_t;
  names_ptr:array of POrtChar;
  ort_values_ptrs:array of POrtValue;
begin
   inputs_num := length(names);
   if length(ort_values)<>inputs_num then
     OrtException.Create('Expecting names and ort_values to have the same length', ORT_INVALID_ARGUMENT);

  setLength(names_ptr,inputs_num);
  setLength(ort_values_ptrs,inputs_num);
  for i := 0 to inputs_num-1 do begin
    names_ptr[i]:=POrtChar(names[i]);
    ort_values_ptrs[i]:=ort_values[i].p_;
  end;
  ThrowOnError(GetApi().AddExternalInitializers(p_, @names_ptr[0], @ort_values_ptrs[0], inputs_num));
  result:=Self

end;

function TORTSessionOptionsHelper.AppendExecutionProvider_CUDA(
  const provider_options: OrtCUDAProviderOptions): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CUDA(p_, @provider_options));
  result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_CUDA_V2(
  const provider_options: OrtCUDAProviderOptionsV2): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(p_, @provider_options));
  Result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_ROCM(
  const provider_options: OrtROCMProviderOptions): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_ROCM(p_, @provider_options));
  Result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_OpenVINO(
  const provider_options: OrtOpenVINOProviderOptions): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_OpenVINO(p_, @provider_options));
  result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_TensorRT(
  const provider_options: OrtTensorRTProviderOptions): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_TensorRT(p_, @provider_options));
  result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_TensorRT_V2(
  const provider_options: OrtTensorRTProviderOptionsV2): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_TensorRT_V2(p_, @provider_options));
  result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_MIGraphX(
  const provider_options: OrtMIGraphXProviderOptions): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_MIGraphX(p_, @provider_options));
  result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider_CANN(
  const provider_options: OrtCANNProviderOptions): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider_CANN(p_, @provider_options));
  Result:=Self
end;

function TORTSessionOptionsHelper.AppendExecutionProvider(
  const provider_name: ortstring; const provider_options: TORTProviderOptions): TORTSessionOptions;
var num_entries,i:size_t;keys,values:array of POrtChar;
begin
  num_entries := provider_options.count;

  if num_entries > 0 then begin
    setLength(keys,num_entries);
    setLength(values,num_entries);

    for i:=0 to num_entries-1 do begin
      keys[i]:=POrtChar(provider_options.keys{$ifndef NO_HASHMAP}.ToArray{$endif}[i]);
      values[i]:=POrtChar(provider_options.values{$ifndef NO_HASHMAP}.ToArray{$endif}[i]);
    end;
  end;
  ThrowOnError(GetApi().SessionOptionsAppendExecutionProvider(p_, POrtChar(provider_name)
                        ,@keys[0], @values[0], num_entries));
  result:=Self;
end;

function TORTSessionOptionsHelper.SetCustomCreateThreadFn(
  ort_custom_create_thread_fn: OrtCustomCreateThreadFn): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsSetCustomCreateThreadFn(p_, ort_custom_create_thread_fn));
  result:=Self
end;

function TORTSessionOptionsHelper.SetCustomThreadCreationOptions(
  ort_custom_thread_creation_options: Pointer): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsSetCustomThreadCreationOptions(p_, ort_custom_thread_creation_options));
  result:=Self
end;

function TORTSessionOptionsHelper.SetCustomJoinThreadFn(
  ort_custom_join_thread_fn: OrtCustomJoinThreadFn): TORTSessionOptions;
begin
  ThrowOnError(GetApi().SessionOptionsSetCustomJoinThreadFn(p_, ort_custom_join_thread_fn));
  result:=Self
end;

{ TORTRunOptionsHelper }

function TORTRunOptionsHelper.SetRunLogVerbosityLevel(level: longint): TORTRunOptions;
begin
  ThrowOnError(GetApi().RunOptionsSetRunLogVerbosityLevel(p_, level));
  result:=self
end;

function TORTRunOptionsHelper.GetRunLogVerbosityLevel: longint;
begin
  ThrowOnError(GetApi().RunOptionsGetRunLogVerbosityLevel(p_, @result));
end;

function TORTRunOptionsHelper.SetRunLogSeverityLevel(level: longint): TORTRunOptions;
begin
  ThrowOnError(GetApi().RunOptionsSetRunLogSeverityLevel(p_, level));
  result:=Self
end;

function TORTRunOptionsHelper.GetRunLogSeverityLevel: longint;
begin
  ThrowOnError(GetApi().RunOptionsGetRunLogSeverityLevel(p_, @result));
end;

function TORTRunOptionsHelper.SetRunTag(const run_tag: POrtChar): TORTRunOptions;
begin
  ThrowOnError(GetApi().RunOptionsSetRunTag(p_, run_tag));
  result:=Self
end;

function TORTRunOptionsHelper.GetRunTag: POrtChar;
begin
  ThrowOnError(GetApi().RunOptionsGetRunTag(p_, @result));
end;

function TORTRunOptionsHelper.AddConfigEntry(const config_key: POrtChar;
  const config_value: POrtChar): TORTRunOptions;
begin
  ThrowOnError(GetApi().AddRunConfigEntry(p_, config_key, config_value));
  result:=Self
end;

function TORTRunOptionsHelper.SetTerminate: TORTRunOptions;
begin
  ThrowOnError(GetApi().RunOptionsSetTerminate(p_));
  result:=Self
end;

function TORTRunOptionsHelper.UnsetTerminate: TORTRunOptions;
begin
  ThrowOnError(GetApi().RunOptionsUnsetTerminate(p_));
  result:=Self
end;

{ TORTCustomOpDomainHelper }

class function TORTCustomOpDomainHelper.Create(const domain: POrtChar):TORTCustomOpDomain;
begin
  ThrowOnError(GetApi().CreateCustomOpDomain(domain, @result.p_));
  result.NewRef();
end;

procedure TORTCustomOpDomainHelper.Add(op: POrtCustomOp);
begin
  ThrowOnError(GetApi().CustomOpDomain_Add(p_, op));
end;

{$ifdef fpc}
{ TORTCustomOpBase }

function TORTCustomOpBase<TOp, TKernel>.CreateKernel(const this_: POrtCustomOp;
  const api: POrtApi; const info: POrtKernelInfo): pointer;
begin
  result:=POp(this_)^.CreateKernel( api^, info);
end;

function TORTCustomOpBase<TOp, TKernel>.GetName(const this_: POrtCustomOp): POrtChar;
begin
  result:=POp(this_)^.GetName;
end;

function TORTCustomOpBase<TOp, TKernel>.GetExecutionProviderType(
  const this_: POrtCustomOp): POrtChar;
begin
  result:=POp(this_)^.GetExecutionProviderType;
end;

function TORTCustomOpBase<TOp, TKernel>.GetInputType(const this_: POrtCustomOp;
  index: size_t): ONNXTensorElementDataType;
begin
  result:=POp(this_)^.GetInputType(index)
end;

function TORTCustomOpBase<TOp, TKernel>.GetInputTypeCount(const this_: POrtCustomOp
  ): size_t;
begin
  result:=POp(this_)^.GetInputTypeCount
end;

function TORTCustomOpBase<TOp, TKernel>.GetOutputType(const this_: POrtCustomOp;
  index: size_t): ONNXTensorElementDataType;
begin
  result:=POp(this_)^.GetOutputType
end;

function TORTCustomOpBase<TOp, TKernel>.GetOutputTypeCount(const this_: POrtCustomOp
  ): size_t;
begin
  result:=POp(this_)^.GetOutputTypeCount
end;

procedure TORTCustomOpBase<TOp, TKernel>.KernelCompute(op_kernel: pointer;
  context: POrtKernelContext);
begin
  PKernel(op_kernel)^.Compute(context)
end;

procedure TORTCustomOpBase<TOp, TKernel>.KernelDestroy(op_kernel: pointer);
begin
  { ToDo : this is maybe wrong equivalent to C++ : delete static_cast<TKernel*>(op_kernel) }
  if assigned(op_kernel) then
    free(PKernel(op_kernel));
end;

function TORTCustomOpBase<TOp, TKernel>.GetInputCharacteristic(const this_: POrtCustomOp; index: size_t): OrtCustomOpInputOutputCharacteristic;
begin
  POp(this_)^.GetInputCharacteristic(index)
end;

function TORTCustomOpBase<TOp, TKernel>.GetOutputCharacteristic(const this_: POrtCustomOp; index: size_t): OrtCustomOpInputOutputCharacteristic;
begin
  POp(this_)^.GetOutputCharacteristic(index)
end;

class operator TORTCustomOpBase<TOp, TKernel>.Initialize(var dest: TORTCustomOpBase<TOp, TKernel>);
begin
  dest.version:=ORT_API_VERSION;
end;

function TORTCustomOpBase<TOp, TKernel>.GetExecutionProviderType: POrtChar;
begin
  result:=nil
end;

function TORTCustomOpBase<TOp, TKernel>.GetInputCharacteristic(index: size_t): OrtCustomOpInputOutputCharacteristic;
begin
  result:=OrtCustomOpInputOutputCharacteristic.INPUT_OUTPUT_REQUIRED
end;

function TORTCustomOpBase<TOp, TKernel>.GetOutputCharacteristic(index: size_t ): OrtCustomOpInputOutputCharacteristic;
begin
  result:=OrtCustomOpInputOutputCharacteristic.INPUT_OUTPUT_REQUIRED;
end;
{$endif}
{ TORTCustomOpApi }

constructor TORTCustomOpApi.Create(const api: OrtApi);
begin
  api_:=api
end;

function TORTCustomOpApi.KernelInfoGetAttribute<T>(const info: POrtKernelInfo;
  const name: POrtChar): T;
begin
  ThrowOnError(api_.KernelInfoGetAttribute_float(info, name, @result));
end;

function TORTCustomOpApi.GetTensorTypeAndShape(const value: POrtValue
  ): POrtTensorTypeAndShapeInfo;
begin
  ThrowOnError(api_.GetTensorTypeAndShape(value, @result))
end;

function TORTCustomOpApi.GetTensorShapeElementCount(
  const info: POrtTensorTypeAndShapeInfo): size_t;
begin
  ThrowOnError(api_.GetTensorShapeElementCount(info, @result));
end;

function TORTCustomOpApi.GetTensorElementType(
  const info: POrtTensorTypeAndShapeInfo): ONNXTensorElementDataType;
begin
  ThrowOnError(api_.GetTensorElementType(info, @result));
end;

function TORTCustomOpApi.GetDimensionsCount(const info: POrtTensorTypeAndShapeInfo
  ): size_t;
begin
  ThrowOnError(api_.GetDimensionsCount(info, @result));
end;

procedure TORTCustomOpApi.GetDimensions(const info: POrtTensorTypeAndShapeInfo;
  dim_values: Pint64_t; dim_values_length: size_t);
begin
  ThrowOnError(api_.GetDimensions(info, dim_values, dim_values_length));
end;

procedure TORTCustomOpApi.SetDimensions(info: POrtTensorTypeAndShapeInfo;
  const dim_values: Pint64_t; dim_count: size_t);
begin
  ThrowOnError(api_.SetDimensions(info, dim_values, dim_count));
end;

function TORTCustomOpApi.GetTensorMutableData<T>(value: POrtValue): Pointer;
begin
  ThrowOnError(api_.GetTensorMutableData(value, @result));
end;

function TORTCustomOpApi.GetTensorData<T>(const value: POrtValue): Pointer;
begin
  result:= GetTensorMutableData<T>(value);
end;

function TORTCustomOpApi.GetTensorMemoryInfo(const value: POrtValue): TOrtMemoryInfo;
begin
  ThrowOnError(api_.GetTensorMemoryInfo(value, @result))
end;

function TORTCustomOpApi.GetTensorShape(const info: POrtTensorTypeAndShapeInfo): TArray<Int64_t>;
begin
  setLength(result,GetDimensionsCount(info));
  GetDimensions(info, @result[0], Length(result));
end;

procedure TORTCustomOpApi.ReleaseTensorTypeAndShapeInfo(input: POrtTensorTypeAndShapeInfo);
begin
  api_.ReleaseTensorTypeAndShapeInfo(input);
end;

function TORTCustomOpApi.KernelContext_GetInputCount(const context: POrtKernelContext): size_t;
begin
  ThrowOnError(api_.KernelContext_GetInputCount(context, @result));
end;

function TORTCustomOpApi.KernelContext_GetInput(const context: POrtKernelContext;index: size_t): POrtValue;
begin
  ThrowOnError(api_.KernelContext_GetInput(context, index, @result));
end;

function TORTCustomOpApi.KernelContext_GetOutputCount(const context: POrtKernelContext): size_t;
begin
  ThrowOnError(api_.KernelContext_GetOutputCount(context, @result));
end;

function TORTCustomOpApi.KernelContext_GetOutput(context: POrtKernelContext; index: size_t; const dim_values: PInt64_t; dim_count: size_t): POrtValue;
begin
  ThrowOnError(api_.KernelContext_GetOutput(context, index, dim_values, dim_count, @result));
end;

function TORTCustomOpApi.KernelContext_GetGPUComputeStream(const context: POrtKernelContext): Pointer;
begin
  ThrowOnError(api_.KernelContext_GetGPUComputeStream(context, @result));
end;

procedure TORTCustomOpApi.ThrowOnError(status: POrtStatus);
begin
  onnxruntime.ThrowOnError(@api_,status);
end;

function TORTCustomOpApi.CreateOpAttr(const name: POrtChar; const data: Pointer; len: longint; _type: OrtOpAttrType): POrtOpAttr;
begin
  ThrowOnError(api_.CreateOpAttr(name, data, len, _type, @result));
end;

procedure TORTCustomOpApi.ReleaseOpAttr(op_attr: POrtOpAttr);
begin
  api_.ReleaseOpAttr(op_attr);
end;

function TORTCustomOpApi.CreateOp(const info: POrtKernelInfo;
  const op_name: POrtChar; const domain: POrtChar; version: longint;
  const type_constraint_names: PPOrtChar;
  const type_constraint_values: PONNXTensorElementDataType;
  type_constraint_count: longint; const attr_values: PPOrtOpAttr;
  attr_count: longint; input_count: longint; output_count: longint): POrtOp;
begin
  ThrowOnError(api_.CreateOp(info, op_name, domain, version, type_constraint_names, type_constraint_values,
                               type_constraint_count, attr_values, attr_count, input_count, output_count, @result));
end;

procedure TORTCustomOpApi.InvokeOp(const context: POrtKernelContext;
  const ort_op: POrtOp; const input_values: PPOrtValue; input_count: longint;
  var output_values: PPOrtValue; output_count: longint);
begin
  ThrowOnError(api_.InvokeOp(context, ort_op, input_values, input_count, output_values, output_count));
end;

procedure TORTCustomOpApi.ReleaseOp(ort_op: POrtOp);
begin
  api_.ReleaseOp(ort_op);
end;

function TORTCustomOpApi.CopyKernelInfo(const info: POrtKernelInfo): POrtKernelInfo;
begin
  ThrowOnError(api_.CopyKernelInfo(info, @result));
end;

procedure TORTCustomOpApi.ReleaseKernelInfo(info_copy: POrtKernelInfo);
begin
  api_.ReleaseKernelInfo(info_copy);
end;

{ TORTAllocatorWithDefaultOptions }

class operator TORTAllocatorWithDefaultOptions.Initialize({$ifdef fpc}var{$else}out{$endif} dest: TORTAllocatorWithDefaultOptions);
begin
  if not Assigned(GetApi()) then exit;
  ThrowOnError(GetApi().GetAllocatorWithDefaultOptions(@dest.p_));
end;

class operator TORTAllocatorWithDefaultOptions.Finalize(var dest: TORTAllocatorWithDefaultOptions);
begin
   //GetApi().ReleaseAllocator(dest.p_);
end;

class operator TORTAllocatorWithDefaultOptions.Implicit(const src: TORTAllocatorWithDefaultOptions): POrtAllocator;
begin
  result:=src.p_;
end;

function TORTAllocatorWithDefaultOptions.Alloc(const size: size_t): Pointer;
begin
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, @result));
end;

function TORTAllocatorWithDefaultOptions.GetAllocation(const size: size_t): TORTMemoryAllocation;
var _out:Pointer;
begin
  ThrowOnError(GetApi().AllocatorAlloc(p_, size, @_out));
  result:=TORTMemoryAllocation.Create(p_, _out, size);
end;

procedure TORTAllocatorWithDefaultOptions.Free(const p: pointer);
begin
  ThrowOnError(GetApi().AllocatorFree(p_, p));
end;

function TORTAllocatorWithDefaultOptions.GetInfo: POrtMemoryInfo;
begin
  ThrowOnError(GetApi().AllocatorGetInfo(p_, @result));
end;

function TORTAllocatorWithDefaultOptions.GetMemoryInfo : TORTMemoryInfo;
begin
  ThrowOnError(GetApi().AllocatorGetInfo(p_, @result.p_));
end;

class operator TORTAllocatorWithDefaultOptions.Implicit(const v:TORTAllocatorWithDefaultOptions):TORTAllocator;
begin
  result.p_:=v.p_
end;

{ TORTMemoryAllocation }

constructor TORTMemoryAllocation.Create(const allocator: POrtAllocator; const p: Pointer; const size: size_t);
begin
  allocator_:=allocator;
  p_:=p;
  size_:=size;
end;

function TORTMemoryAllocation.get: pointer;
begin
  result:=p_
end;

function TORTMemoryAllocation.size: size_t;
begin
  result:=size_
end;

class operator TORTMemoryAllocation.Finalize(var dest: TORTMemoryAllocation);
//var ret:POrtStatus;
begin
  if assigned(dest.p_) then begin
    // We do not throw out of destructor
    {ret :=} GetApi().AllocatorFree(dest.allocator_, dest.p_);
  end;
end;

{ TORTEnvHelper }

class function TORTEnvHelper.Create(logging_level: OrtLoggingLevel; const logid: POrtChar):TORTEnv;
begin
  ThrowOnError(GetApi().CreateEnv(logging_level, logid, @result.p_));
  ThrowOnError(GetApi().SetLanguageProjection(@result.p_, DefaultLanguageProjection));
  if Assigned(DefaultEnv.p_) then result.NewRef
end;

class function TORTEnvHelper.Create(logging_level: OrtLoggingLevel;
  const logid: POrtChar; logging_function: OrtLoggingFunction;
  logger_param: Pointer):TORTEnv;
begin
  ThrowOnError(GetApi().CreateEnvWithCustomLogger(logging_function, logger_param, logging_level, logid, @result.p_));
  ThrowOnError(GetApi().SetLanguageProjection(@result.p_, DefaultLanguageProjection));
  result.NewRef
end;

class function TORTEnvHelper.Create(const tp_options: POrtThreadingOptions;
  logging_level: OrtLoggingLevel; const logid: POrtChar):TORTEnv;
begin
  ThrowOnError(GetApi().CreateEnvWithGlobalThreadPools(logging_level, logid, tp_options, @result.p_));
  ThrowOnError(GetApi().SetLanguageProjection(@result.p_, DefaultLanguageProjection));
  result.NewRef
end;

class function TORTEnvHelper.Create(const tp_options: POrtThreadingOptions;
  logging_function: OrtLoggingFunction; logger_param: Pointer;
  logging_level: OrtLoggingLevel; const logid: POrtChar):TORTEnv;
begin
  ThrowOnError(GetApi().CreateEnvWithCustomLoggerAndGlobalThreadPools(logging_function, logger_param, logging_level, logid, tp_options, @result.p_));
  ThrowOnError(GetApi().SetLanguageProjection(@result.p_, DefaultLanguageProjection));
  result.NewRef
end;

function TORTEnvHelper.EnableTelemetryEvents: TORTEnv;
begin
  ThrowOnError(GetApi().EnableTelemetryEvents(p_));
  result:=Self.p_;
end;

function TORTEnvHelper.DisableTelemetryEvents:TORTEnv;
begin
  ThrowOnError(GetApi().DisableTelemetryEvents(p_));
  result:=Self.p_
end;

function TORTEnvHelper.CreateAndRegisterAllocator(const mem_info: TOrtMemoryInfo;
  const arena_cfg: TOrtArenaCfg): TORTEnv;
begin
  ThrowOnError(GetApi().CreateAndRegisterAllocator(p_, mem_info.p_, arena_cfg.p_));
  result:=Self.p_
end;

function TORTEnvHelper.UpdateLogLevel(const log_level: OrtLoggingLevel): TORTEnv;
begin
  ThrowOnError(GetApi().UpdateEnvWithCustomLogLevel(p_, log_level));
  result:=Self.p_;
end;


{ BFloat16_t }

constructor BFloat16_t.Create(const v: uint16_t);
begin
  value:=v
end;

class operator BFloat16_t.Implicit(const v: BFloat16_t): uint16_t;
begin
  result:=v.value
end;

class operator BFloat16_t.Implicit(const v: uint16_t): BFloat16_t;
begin
  result.value:=v
end;

{ Float16_t }

constructor Float16_t.Create(const v: uint16_t);
begin
  value:=v
end;

class operator Float16_t.Implicit(const v: Float16_t): uint16_t;
begin
  result:=v.value
end;

class operator Float16_t.Implicit(const v: uint16_t): Float16_t;
begin
  result.value:=v
end;

{ OrtException }

function OrtException.what: ortstring;
begin
  result:=Message;
end;

constructor OrtException.Create(const str: ortstring; code: OrtErrorCode);
begin
  code_:=code;
  inherited CreateFmt('Code [%d]: %s',[Ord(code),str]);
end;


initialization


  DefaultEnv:=TORTEnv.Create(ORT_LOGGING_LEVEL_WARNING, POrtChar(DEFAULT_LOGID) );

  //if not assigned(DefaultSessionOptions.p_) then begin
  //  ThrowOnError(Api.CreateSessionOptions(@DefaultSessionOptions.p_));
  //  DefaultSessionOptions.NewRef();
  //end ;
  //
  //if not Assigned(DefaultRunOptions.p_) then begin
  //  ThrowOnError(Api.CreateRunOptions(@DefaultRunOptions.p_));
  //  DefaultRunOptions.NewRef();
  //end
finalization
  GetApi().ReleaseRunOptions(DefaultRunOptions.p_);
  GetApi().ReleaseSessionOptions(DefaultSessionOptions.p_);
  GetApi().ReleaseEnv(DefaultEnv.p_);

  if Assigned(@HouseKeeper) then
    if IsConsole then writeLn('clear');


end.
