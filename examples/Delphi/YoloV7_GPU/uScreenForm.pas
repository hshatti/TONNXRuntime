unit uScreenForm;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes,
  FMX.Graphics,FMX.Forms, FMX.Memo,  System.Generics.Collections,System.Generics.Defaults, System.Math,
  onnxruntime,onnxruntime.dml, onnxruntime_pas_api, FMX.Memo.Types, FMX.Types, FMX.Controls,
  FMX.Controls.Presentation, FMX.ScrollBox,StrUtils;


 const
    enable_nms : Boolean = True;

// for 640x640x3 Yolov7 MS COCO
    D1 = 25200;
    NC = 80;
    D2 = NC+5;
    MODEL_PATH = 'nms_yolov7_25200.onnx';
    score_threshold=0.1;
    nms_threshold=0.5;
    DET_W = 640;  // detector input width, height and depth
    DET_H = 640;
    DET_D = 3;

//const // for 1280x1280x3 Yolov7-w6 MS COCO
//    D1 = 102000;
//    NC = 80;
//    D2 = NC+5;
//    MODEL_PATH = 'yolov7-w6.onnx';
//    score_threshold=0.1;
//    nms_threshold=0.5;
//    DET_W = 1280;  // detector input width, height and depth
//    DET_H = 1280;
//    DET_D = 3;
type
  PDetBox=^TDetBox;
  TDetBox=record
    bbox   : array [0..3] of Single; // x1 y1 x2 y2
    scores : array [0..NC-1] of Single; // confidences by class
    conf   : Single;  // max of scores    - detection confidence
    cls    : Integer; // argmax of scores - detection class
    area   : Single;  // bbox area        - for nms
  end;

  TDetBoxList=class(TList)
  private
    function Get(Index: Integer): PDetBox;
  public
    destructor Destroy; override;
    function Add(Value: PDetBox): Integer;
    property Items[Index: Integer]: PDetBox read Get; default;
  end;

  TScreenForm = class(TForm)
    Memo1: TMemo;
    procedure FormCreate(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  ScreenForm: TScreenForm;
// ONNX model variables
  ModelPath:string;
  session:TORTSession;

  input:TOrtTensor<single>;
  outTensor: TOrtTensor<single>;

  thresholded_cnt: Integer;  // Count of all thresholded detections

  nmsed_cnt: Integer;        // Count of all detections before NMS

  det_scores: array of array of Single;  // final scores matrix (nmsed_cntxNC)
  det_boxes: array of array of Single;   // final boxes matrix  (nmsed_cntx4)
  det_classes: array of Integer;         // final class vector  (nmsed_cnt) (index of argmax on scores row)
  det_argmax_score: array of Single;     // final scores vector (nmsed_cnt) (max on scores row)
  det_areas: array of Single;            // final areas vector  (nmsed_cnt)
  det_cnt: Integer;                      // Count of all detections after NMS
  det_running: Boolean;
  det_inference_time: size_t;           // inference time in ms
  det_nms_time: size_t;                 // nms time in ms

var
  InputNames,OutputNames:TArray<ansistring>;
  Info          :TORTTypeInfo;
  MetaData      :TORTModelMetadata;
  version       :Int64_t;
  memInfo       :TORTMemoryInfo;
  s             :ansistring;
  memtype       :OrtMemType;
  alloctype     :OrtAllocatorType;
  deviceid      :size_t;
//  sessionOptions:TORTSessionOptions;

  providerOptionsCUDAV2 : OrtCUDAProviderOptionsV2;


  procedure InitONNXRuntime(Sender: TObject);
  function InferenceFromBitmap(const bmp_in:TBitmap; enable_nms:Boolean):Integer;
  procedure DetReset();

implementation

{$R *.fmx}


{ TDetBoxList }

function TDetBoxList.Add(Value: PDetBox): Integer;
begin
  Result := inherited Add(Value);
end;

destructor TDetBoxList.Destroy;
var
  i: Integer;
begin
  for i := 0 to Count - 1 do
    Dispose(Items[i]);

  inherited;
end;

function TDetBoxList.Get(Index: Integer): PDetBox;
begin
  Result := PDetBox(inherited Get(Index));
end;

function DetBoxCompareByConf(const d1, d2: TDetBox): Integer;
begin
  if d1.conf	 = d2.conf then
    result := 0
  else if d1.conf < d2.conf then
    result := 1
  else
    result := -1;
end;

procedure  WriteLog(str:ansistring);
begin
  ScreenForm.Memo1.lines.add(str);
end;

procedure  ClearLog();
begin
  ScreenForm.Memo1.lines.Clear;
end;

function Join(const arr:array of ansistring; const Delimiter:ansistring=', '):ansistring;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+arr[i];
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

function Join(const arr:array of int64_t; const Delimiter:ansistring=', '):ansistring;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+IntToStr(arr[i]);
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

function Join(const arr:array of double; const Delimiter:ansistring=', '):ansistring;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+FloatToStr(arr[i]);
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

function TypeInfoDesc(const _type:TOrtTypeInfo):ansistring;
var i:size_t;
  tensorInfo:TORTTensorTypeAndShapeInfo;
  sequenceInfo:TORTSequenceTypeInfo;
  mapInfo:TORTMapTypeInfo;
  strs:TArray<ansistring>;ints:TArray<int64_t>;
begin
  result:='';
  case _type.GetOnnxType() of
    ONNX_TYPE_UNKNOWN:
      result:='Unknown';
    ONNX_TYPE_TENSOR: begin
      tensorInfo:=_type.GetTensorTypeAndShapeInfo();
      result:=ORTTensorTypes[tensorInfo.GetElementType]+'[';
      ints:=tensorInfo.GetShape();
      result:=result+Join(ints,',')+']';
      //result:=result+' #'+Join(TensorInfo.GetSymbolicDimensions(length(ints)));
    end;
    ONNX_TYPE_SEQUENCE:begin
      result:='Sequence(';
      sequenceInfo:=_type.GetSequenceTypeInfo();
      result:=result+TypeInfoDesc(sequenceInfo.GetSequenceElementType())+')';
    end;
    ONNX_TYPE_MAP:begin
      result:='Map';

    end;
    ONNX_TYPE_OPAQUE:
      result:='Opaque';
    ONNX_TYPE_SPARSETENSOR:begin
      result:='SparseTensor';
      //tensorInfo:=_type.GetTensorTypeAndShapeInfo();
    end;
    ONNX_TYPE_OPTIONAL:
      result:='Optional'
  end;
end;


procedure InitONNXRuntime(Sender: TObject);
var
  i : Integer;
  cudaOpt:PORTChar;
  cpok,cpov:array of PORTChar;
  cpo:POrtCUDAProviderOptionsV2;
//  so:TORTSessionOptions;
  DMLprov:POrtDmlApi;
begin


//    raise Exception.create('cannot initialize COM lib.');
//  writeln(format('%x',[CoInitializeEx(nil,COINIT_MULTITHREADED)]));
  ClearLog();

  WriteLog('Init ORT Engine');
  ModelPath:=MODEL_PATH;

  WriteLog('------------------ Start ----------------');
  WriteLog('Providers :'+ join(GetAvailableProviders));


//  so:=DefaultSessionOptions.Clone;
//  setLength(cpok,1);
//  setLength(cpov,1);
//  cpok[0]:= 'device_id';
//  cpov[0]:='0';
//  ThrowOnError(getapi().CreateCUDAProviderOptions(@cpo));
//  ThrowOnError(getapi().UpdateCUDAProviderOptions(@cpo,@cpok[0],@cpov[0],length(cpok)));
//  ThrowOnError(getapi().SessionOptionsAppendExecutionProvider_CUDA_V2(DefaultSessionOptions.p_,@cpo));

  // DML : Session option must run as a single execution with no memory patterns as per ORT documentation
  DefaultSessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
  DefaultSessionOptions.DisableMemPattern();
  ThrowOnError(GetApi().GetExecutionProviderApi('DML',ORT_API_VERSION,@DMLprov));
  // DML : if Primary device is Intel Graphics DML may take some time to load the session,
  // NVIDIA seems to lead faster, try to change the deviceId if so.
  ThrowOnError(DMLProv.SessionOptionsAppendExecutionProvider_DML(DefaultSessionOptions.p_,0));

  WriteLog(format('Loading Model [%s]...',[ModelPath]));

  session:=TORTSession.Create(Modelpath);
  WriteLog('Model Loaded.');

  setLength(InputNames,Session.GetInputCount());
  for i:=0 to High(InputNames) do begin
    Info:=Session.GetInputTypeInfo(i);
    InputNames[i]:=ansistring(session.GetInputNameAllocated(i,DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif})+' :'+TypeInfoDesc(Info);
//    writeln('Input > ',InputNames[i])
  end;
  setLength(OutputNames,Session.GetOutputCount());
  for i:=0 to High(OutputNames) do begin
    Info:=Session.GetOutputTypeInfo(i);
    OutputNames[i]:=ansistring(session.GetOutputNameAllocated(i,DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif})+' :'+TypeInfoDesc(Info);
//    writeln('Output > ',OutputNames[i]);
  end;

  MetaData:=session.GetModelMetadata;
  WriteLog('Description : '+ansistring(Metadata.GetDescriptionAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif}));
  WriteLog('Producer : '   +ansistring(MetaData.GetProducerNameAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif}));
  WriteLog('Graph Name : ' +ansistring(MetaData.GetGraphNameAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.instance{$endif}));
  WriteLog('Graph Description : '+ansistring(MetaData.GetGraphDescriptionAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.instance{$endif}));
  WriteLog('Domain : '+ansistring(MetaData.GetDomainAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.instance{$endif}));
  //version:=MetaData.GetVersion();
  //WriteLog('Version : '+FloatToStr(Version));
  WriteLog(format(' Inputs => Outputs : [%s] => [%s]',[Join(InputNames), Join(OutputNames)]));
  WriteLog('------------------- End ----------------');
  MemInfo:=DefaultAllocator.GetMemoryInfo;
  s:=MemInfo.GetAllocatorName();

  WriteLog('Memtype :'      +IntToStr(Ord(MemInfo.GetMemoryType())   ));
  WriteLog('AllocatorType :'+IntToStr(Ord(MemInfo.GetAllocatorType())));
  // it seems that Ort does not necessarily return the inference device, just the current memory device
  WriteLog('Device Type : '+ifthen(MemInfo.GetDeviceType=OrtMemoryInfoDeviceType_CPU,'CPU','GPU'));
  WriteLog('DeviceId :'     +IntToStr(Ord(MemInfo.GetDeviceId())     ));
  WriteLog('==============');

end;

procedure TScreenForm.FormCreate(Sender: TObject);
begin
   InitONNXRuntime(Sender);
   det_running:=False;
end;


// resizes by long side
procedure ResizeImage(const Src:TBitmap; var Dst: TBitmap; const OutWidth, OutHeight, DstWidth, DstHeight: Integer);
begin
  try
    Dst := TBitmap.Create;
    Dst.SetSize(DstWidth, DstHeight);
    Dst.Canvas.BeginScene();
    Dst.Canvas.DrawBitmap(Src ,Rect(0,0,Src.Width, Src.Height),Rect(0,0,Round(OutWidth),Round(OutHeight)),255);
    Dst.Canvas.EndScene;
  except on E:Exception do
    begin
      FreeAndNil(Dst)    ;
      raise E;
    end;
  end;
end;

procedure DetReset();
begin
  det_cnt:=0;
  SetLength(det_scores,      det_cnt, NC);
  SetLength(det_boxes,       det_cnt, 4);
  SetLength(det_classes,     det_cnt);
  SetLength(det_argmax_score,det_cnt);
  SetLength(det_areas,       det_cnt);
end;

function InferenceFromBitmap(const bmp_in:TBitmap; enable_nms:Boolean):Integer;
var
    t1,t2,t3:size_t;
    y,x:int64;
    i,j,k: Integer;
    inSize, OutSize:TSize;
    TensorDim:TArray<int64_t>;
    pixelSpan:TBitmapData;
    inputs,outputs:TORTNameValueList;
    img : TBitmap;

    x_box, y_box, h_box, w_box, conf_scale: Single;
    max_score: Single;
    argmax_class : Integer;
    scale_factor_x,scale_factor_y: Single;


    x1_0, y1_0, x2_0, y2_0,conf_0,area_0: Single;
    x1_p, y1_p, x2_p, y2_p,conf_p,area_p: Single;
    xx1, yy1, xx2, yy2: Single;
    wp, hp: Single;
    inter, overlap: Single;
    cls_0, cls_p: Integer;
    //
    ThresholdedDetBoxList: TDetBoxList;
    DB:PDetBox;
    DetBox: TDetBox;
    NmsedDetBoxList: TDetBoxList;
    discard_box: Boolean;
    IN_W, IN_H, OUT_W,OUT_H: Integer;

begin
  IN_W  := bmp_in.Canvas.Width;
  IN_H  := bmp_in.Canvas.Height;

  OUT_W := DET_W; // resized for inference image width and height (width is always long side)
  OUT_H := Round(IN_H/(IN_W/DET_W)); // scale height accordingly

  det_inference_time:=0;
  // init global variables
  DetReset();
  Result:=-1;
  det_running:=True;

  try
    ResizeImage(bmp_in,img,OUT_W,OUT_H, DET_W,DET_H);
  except
    det_running:=False;
    Result:=-3;
    Exit();
  end;

  Input:=  TOrtTensor<single>.Create([DET_D,DET_W,DET_H]);
  try
    img.Map(TMapAccess.Read,pixelSpan);
    for  y:= 0 to img.height-1 do begin
      for x := 0 to img.width-1 do begin
          input[0,x,y] := single(TAlphaColorRec(pixelSpan.GetPixel(x,y)).B);
          input[1,x,y] := single(TAlphaColorRec(pixelSpan.GetPixel(x,y)).G);
          input[2,x,y] := single(TAlphaColorRec(pixelSpan.GetPixel(x,y)).R);
      end
    end;
    img.Unmap(pixelSpan);
  except
    det_running:=False;
    Result:=-4;
    Exit;
  end;
  img.Free;
  t1:=TThread.GetTickCount;
  try
    inputs.AddOrSetValue('images',input);
    Outputs:=session.Run(Inputs);
    outTensor := outputs['output'];
  except
    det_running:=False;
    Result:=-5;
    Exit;
  end;
  t1:=TThread.GetTickCount-t1;

  t2:=TThread.GetTickCount;

  // rewriting to List of TDetBox
  ThresholdedDetBoxList:= TDetBoxList.Create;
  for i := 0 to outTensor.Shape[1]-1 do
  begin
//    writeln(outTensor.Shape[0],', ',outTensor.Shape[1],', ',outTensor.Shape[2]);
    // assign coordinates
    x_box  := outTensor[0,i,0]; // predictions[i,0]; x1
    y_box  := outTensor[0,i,1]; // predictions[i,1]; y1
    w_box  := outTensor[0,i,2]; // predictions[i,2]; x2
    h_box  := outTensor[0,i,3]; // predictions[i,3]; y2
    conf_scale   := outTensor[0,i,4]; // predictions[i,4]; conf_scale
    // convert bboxes
    scale_factor_x:=IN_W/OUT_W; // resolution is fixed in constants for now, remake later
    scale_factor_y:=IN_H/OUT_H;

    DetBox.bbox[0] := (x_box - w_box*0.5)*scale_factor_x;
    DetBox.bbox[1] := (y_box - h_box*0.5)*scale_factor_y;
    DetBox.bbox[2] := (x_box + w_box*0.5)*scale_factor_x;
    DetBox.bbox[3] := (y_box + h_box*0.5)*scale_factor_y;
    // convert scores
    for j:=0 to NC-1 do
    begin
      DetBox.scores[j]:= conf_scale*outTensor[0,i,j+5]; //conf[i]*predictions[i,j+5];
    end;
    max_score:=0;
    argmax_class:=0;
    // argmax to get max_score (before thesholding)
    for j:=0 to NC-1 do
    begin
      if DetBox.scores[j]> max_score then
      begin
         max_score:=DetBox.scores[j];
         argmax_class:=j;
      end;
    end;

    if (max_score)> score_threshold then
    begin
      // correct boxes cooordinates if they are outside the input image dimensions
      // correct x1 x2
      New(DB);
      if DetBox.bbox[0] <0 then
         DetBox.bbox[0]:=0;
      if DetBox.bbox[0] >=IN_W-1 then
         DetBox.bbox[0]:=IN_W-1;
      if DetBox.bbox[2] <0 then
         DetBox.bbox[2]:=0;
      if DetBox.bbox[2] >IN_W-1 then
         DetBox.bbox[2]:=IN_W-1;
      // correct y1 y2
      if DetBox.bbox[1] <0 then
         DetBox.bbox[1]:=0;
      if DetBox.bbox[1] >=IN_H-1 then
         DetBox.bbox[1]:=IN_H-1;
      if DetBox.bbox[3] <0 then
         DetBox.bbox[3]:=0;
      if DetBox.bbox[3] >IN_H-1 then
         DetBox.bbox[3]:=IN_H-1;
      DetBox.area:= (DetBox.bbox[3]-DetBox.bbox[1]+1)*
                    (DetBox.bbox[2]-DetBox.bbox[0]+1);
      DetBox.conf:= max_score;
      DetBox.cls := argmax_class;
      Move(DetBox, DB^, SizeOf(TDetBox));
      ThresholdedDetBoxList.Add(DB);
    end;
  end;
  thresholded_cnt:= ThresholdedDetBoxList.Count ;
//  FreeAndNil(ThresholdedDetBoxList);exit;

// if no detections - normal exit;
  if thresholded_cnt=0 then begin
     Result:=0;
     det_running:=False;
     ThresholdedDetBoxList.Free; //free memory
     Exit;
  end;

  // sort list of DetBox by conf
  ThresholdedDetBoxList.Sort(
    @DetBoxCompareByConf
  );

// if enable_nms = False - return only thresholded detections
  if not(enable_nms) then
  begin
    setLength(det_scores, thresholded_cnt, NC);
    setLength(det_boxes, thresholded_cnt, 4);
    setLength(det_classes, thresholded_cnt);
    setLength(det_argmax_score, thresholded_cnt);
    setLength(det_areas, thresholded_cnt);
    det_cnt := 0;
    for i := 0 to thresholded_cnt - 1 do
    begin
        det_boxes[det_cnt, 0] := ThresholdedDetBoxList[i].bbox[0];
        det_boxes[det_cnt, 1] := ThresholdedDetBoxList[i].bbox[1];
        det_boxes[det_cnt, 2] := ThresholdedDetBoxList[i].bbox[2];
        det_boxes[det_cnt, 3] := ThresholdedDetBoxList[i].bbox[3];
        for k := 0 to NC - 1 do
        begin
          det_scores[det_cnt, k] := ThresholdedDetBoxList[i].scores[k];
        end;
        det_areas[det_cnt] := ThresholdedDetBoxList[i].area;
        det_classes[det_cnt] := ThresholdedDetBoxList[i].cls;
        det_argmax_score[det_cnt] := ThresholdedDetBoxList[i].conf;
        det_cnt := det_cnt + 1;
    end;
    result := 0;
    det_running := False;
    ThresholdedDetBoxList.Free; //free memory
    Exit();
  end;

// get here only if enable_nms = True

  // rewritten to NmsedDetBoxList
  NmsedDetBoxList := TDetBoxList.Create;
  // NMS thresholded predictions,
  for i := 0 to thresholded_cnt - 1 do
  begin
    discard_box := False;
    cls_0 := ThresholdedDetBoxList.Items[i].cls;
    x1_0 := ThresholdedDetBoxList.Items[i].bbox[0];
    y1_0 := ThresholdedDetBoxList.Items[i].bbox[1];
    x2_0 := ThresholdedDetBoxList.Items[i].bbox[2];
    y2_0 := ThresholdedDetBoxList.Items[i].bbox[3];
    area_0 := ThresholdedDetBoxList.Items[i].area;
    conf_0 := ThresholdedDetBoxList.Items[i].conf;
    for j := 0 to thresholded_cnt - 1 do
    begin
      cls_p := ThresholdedDetBoxList.Items[j].cls;
      x1_p  := ThresholdedDetBoxList.Items[j].bbox[0];
      y1_p  := ThresholdedDetBoxList.Items[j].bbox[1];
      x2_p  := ThresholdedDetBoxList.Items[j].bbox[2];
      y2_p  := ThresholdedDetBoxList.Items[j].bbox[3];
      area_p := ThresholdedDetBoxList.Items[j].area;
      conf_p := ThresholdedDetBoxList.Items[j].conf;
      if cls_0 = cls_p  then
      begin
        xx1 := Max(x1_0, x1_p);
        yy1 := Max(y1_0, y1_p);
        xx2 := Min(x2_0, x2_p);
        yy2 := Min(y2_0, y2_p);
        wp := xx2 - xx1 + 1;
        hp := yy2 - yy1 + 1;
        if ((wp > 0) and (hp > 0)) then
        begin
          inter := wp * hp;
          overlap := inter / (area_p + area_0 - inter);
          if (overlap > nms_threshold) then
          begin
            if conf_p > conf_0 then
              discard_box := True;
          end;
        end;
        if discard_box then
           Break;
      end;
    end;
    if not(discard_box) then
    begin
      NmsedDetBoxList.Add(ThresholdedDetBoxList.Items[i]);
    end;
  end;
  nmsed_cnt := NmsedDetBoxList.Count;

  SetLength(det_scores,      nmsed_cnt, NC);
  SetLength(det_boxes,       nmsed_cnt, 4);
  SetLength(det_classes,     nmsed_cnt);
  SetLength(det_argmax_score,nmsed_cnt);
  SetLength(det_areas,       nmsed_cnt);
  det_cnt:=0;
  for i:=0 to nmsed_cnt-1 do begin
     det_boxes[det_cnt,0]:=NmsedDetBoxList.Items[i].bbox[0];
     det_boxes[det_cnt,1]:=NmsedDetBoxList.Items[i].bbox[1];
     det_boxes[det_cnt,2]:=NmsedDetBoxList.Items[i].bbox[2];
     det_boxes[det_cnt,3]:=NmsedDetBoxList.Items[i].bbox[3];
     for k:=0 to NC-1 do
     begin
          det_scores[det_cnt,k] :=NmsedDetBoxList.Items[i].scores[k];
     end;
     det_areas[det_cnt]  :=NmsedDetBoxList.Items[i].area;
     det_classes[det_cnt]:=NmsedDetBoxList.Items[i].cls	;
     det_argmax_score[det_cnt]:=NmsedDetBoxList.Items[i].conf;
     det_cnt:=det_cnt+1;
  end;
  NmsedDetBoxList.Free;
  t2:=TThread.GetTickCount-t2;
  Result:=0;
  det_inference_time := t1;
  det_nms_time := t2;
  det_running:=False;
  Exit();
end;

end.
