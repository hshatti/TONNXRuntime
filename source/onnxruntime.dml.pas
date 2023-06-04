unit onnxruntime.dml;


interface
uses onnxruntime_pas_api;
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


type
  POrtDmlApi = ^OrtDmlApi;
  OrtDmlApi = record
    (**
     * Creates a DirectML Execution Provider which executes on the hardware adapter with the given device_id, also known as
     * the adapter index. The device ID corresponds to the enumeration order of hardware adapters as given by
     * IDXGIFactory::EnumAdapters. A device_id of 0 always corresponds to the default adapter, which is typically the
     * primary display GPU installed on the system. A negative device_id is invalid.
     *)
    SessionOptionsAppendExecutionProvider_DML:function(const options:POrtSessionOptions;  device_id:longint):POrtStatus; stdcall;

    (**
     * Creates a DirectML Execution Provider using the given DirectML device, and which executes work on the supplied D3D12
     * command queue. The DirectML device and D3D12 command queue must have the same parent ID3D12Device, or an error will
     * be returned. The D3D12 command queue must be of type DIRECT or COMPUTE (see D3D12_COMMAND_LIST_TYPE). If this
     * function succeeds, the inference session maintains a strong reference on both the dml_device and the command_queue
     * objects.
     * See also: DMLCreateDevice
     * See also: ID3D12Device::CreateCommandQueue
     *)
    SessionOptionsAppendExecutionProvider_DML1:function(const options: POrtSessionOptions ;
                    dml_device: Pointer {IDMLDevice*} ;  cmd_queue:Pointer{ID3D12CommandQueue*}):POrtStatus; stdcall;

    (**
     * CreateGPUAllocationFromD3DResource
     * This API creates a DML EP resource based on a user-specified D3D12 resource.
     *)
    CreateGPUAllocationFromD3DResource:function(const d3d_resource:pointer {ID3D12Resource*} ;  dml_resource:PPointer ):POrtStatus;stdcall;

    (**
     * FreeGPUAllocation
     * This API frees the DML EP resource created by CreateGPUAllocationFromD3DResource.
     *)
    FreeGPUAllocation:function(const dml_resource: Pointer ):POrtStatus;stdcall;

    (**
     * GetD3D12ResourceFromAllocation
     * This API gets the D3D12 resource when an OrtValue has been allocated by the DML EP.
     *)
    GetD3D12ResourceFromAllocation:function(const provider :POrtAllocator;const dml_resource:Pointer; d3d_resource:PPointer { ID3D12Resource**} ):POrtStatus;stdcall;
  end;
implementation

end.
