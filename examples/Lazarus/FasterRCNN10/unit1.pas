unit Unit1;

{$mode delphi}{$H+}

{$ifdef fpc}
{$PACKRECORDS 32}
  {$ifdef CPUX86_64}
    {$asmmode intel}
  {$endif}
  {$PointerMath On}
{$endif}

{.$define NO_SMARTPTR}

interface

uses
  Types, Classes, SysUtils, strutils, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Buttons, onnxruntime_pas_api, onnxruntime, Math
  ,Generics.Collections
  ;

type

  { TForm1 }

  TForm1 = class(TForm)
    BitBtn1: TBitBtn;
    BitBtn2: TBitBtn;
    CheckBox1: TCheckBox;
    ComboBox1: TComboBox;
    Image1: TImage;
    Image2: TImage;
    Label1: TLabel;
    Label2: TLabel;
    Memo1: TMemo;
    OpenDlg: TOpenDialog;
    Panel1: TPanel;
    Panel2: TPanel;
    SaveDlg: TSaveDialog;
    ScrollBox1: TScrollBox;
    ScrollBox2: TScrollBox;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    procedure BitBtn2Click(Sender: TObject);
    procedure WriteLog(const str:string);
    procedure BitBtn1Click(Sender: TObject);
    procedure CheckBox1Change(Sender: TObject);
    procedure ComboBox1Change(Sender: TObject);
    procedure FormShow(Sender: TObject);
  private

  public

  end;

  TScaleType =(stNearest, stLinear);
  PBGRA = ^ TBGRA;
  TBGRA = packed record
    case byte of
      0 :(b,g,r,a:byte);
      1 :(v:array[0..3] of byte);
      2 :(i:longword)
  end;
  PBGR = ^ TBGR;
  TBGR = record
    case byte of
      0:(b,g,r:byte);
      1:(v:array[0..2] of byte);
  end;

  const _labels:array[0..80] of string=(
  '__background',
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush'
  );

type
  TPredict=record
  type
     TBox = record
       xmin,ymin,xmax,ymax:single;
     end;
  public
    box:TBox;
    _Label:string;
    confidence:single
  end;


var


  Form1: TForm1;
  ModelPath:string ;
  session:TORTSession;
  input:TORTValue;

  function Join(const arr:array of string; const Delimiter:string=', '):string;  overload;
  function Join(const arr:array of int64_t; const Delimiter:string=', '):string;  overload;
  function Join(const arr:array of double; const Delimiter:string=', '):string;  overload;
  //function TypeName(const _type:ONNXTensorElementDataType):string;
  function ORTTypeInfoDesc(const _type:TOrtTypeInfo):string;

implementation

{$R *.lfm}

procedure ScaleImage32<T>(const inSize,outSize:TSize;const inData, outData:T;scaleType:TScaleType=stNearest);
var i,j,i1,j2:integer;p1,p2:TBGRA; pDst,pSrc,pSrc2:PBGRA; tx,ty:single;
function lerp(const a,b:TBGRA;const t:single):TBGRA; inline;

begin

  result.v[0]:=round(a.v[0]+(b.v[0]-a.v[0])*t);
  result.v[1]:=round(a.v[1]+(b.v[1]-a.v[1])*t);
  result.v[2]:=round(a.v[2]+(b.v[2]-a.v[2])*t);
  //result.v[3]:=round(a.v[3]+(b.v[3]-a.v[3])*t);
end;

var xRatio,yRatio:single;
begin
  outData.beginUpdate;
  xRatio:=inSize.cx/outSize.cx;
  yRatio:=inSize.cy/outSize.cy;
  if scaleType=stNearest then begin
    for i:=0 to outsize.cy-1 do begin
      //pDst:=outData.scanline[i];
      //pSrc:=inData.ScanLine[round(i*yRatio)];
      for j:=0 to outsize.cx-1 do begin
        //pDst^:=pSrc[round(j*xRatio)];
        //inc(pDst);
        outData.Canvas.pixels[j,i]:=inData.canvas.pixels[round(j*xRatio),round(i*yRatio)]
      end;
    end;
    exit
  end;
  for i:=0 to outSize.cy-2 do begin
    pDst:=outData.scanline[i];
    pSrc:=inData.ScanLine[round(i*yRatio)];
    pSrc2:=inData.ScanLine[round((i+1)*yRatio)];
    ty:=frac(i*yRatio);
    for j:=0 to outsize.cx-1 do begin
      tx:=frac(j*xRatio);
      p1:=lerp( pSrc[round(j*xRatio)],  pSrc[round((j+1)*xRatio)],tx);
      p2:=lerp(pSrc2[round(j*xRatio)], pSrc2[round((j+1)*xRatio)],tx);
      pDst^:=lerp(p1,p2,ty);
      inc(pDst);
      pDst^:=lerp(p1,p2,ty);
    end;
  end;
  pSrc:=inData.ScanLine[round((outSize.cy-1)*yRatio)];
  for j:=0 to outsize.cx-1 do
    pDst[j]:=pSrc[round(j*xRatio)];
    //outData.Canvas.Pixels[j,outSize.cy-1]:=pSrc[round(j*xRatio)].i;
  outData.endUpdate

end;

{ TForm1 }
function Join(const arr:array of string; const Delimiter:string=', '):string;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+arr[i];
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

function Join(const arr:array of int64_t; const Delimiter:string=', '):string;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+IntToStr(arr[i]);
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

function Join(const arr:array of double; const Delimiter:string=', '):string;  overload;
var i:integer;
begin
  result:='';
  for i:=0 to high(arr) do
    result:=result+Delimiter+FloatToStr(arr[i]);
  if length(result)>0 then
    delete(result,1,length(Delimiter))
end;

procedure TForm1.CheckBox1Change(Sender: TObject);
begin
  Image1.Stretch:=Checkbox1.Checked;
  Image1.StretchInEnabled:=Checkbox1.Checked;
  Image1.StretchOutEnabled:=Checkbox1.Checked;
  Image1.AutoSize:=not Checkbox1.Checked;
  Image1.SetBounds(0,0,ScrollBox1.Width,Scrollbox1.Height);
  Scrollbox1.HorzScrollBar.Range:=0;
  Scrollbox1.VertScrollBar.Range:=0;

  Image2.Stretch:=Checkbox1.Checked;
  Image2.StretchInEnabled:=Checkbox1.Checked;
  Image2.StretchOutEnabled:=Checkbox1.Checked;
  Image2.AutoSize:=not Checkbox1.Checked;
  Image2.SetBounds(0,0,ScrollBox2.Width,Scrollbox2.Height);
  Scrollbox2.HorzScrollBar.Range:=0;
  Scrollbox2.VertScrollBar.Range:=0;
end;


function ORTTypeInfoDesc(const _type:TOrtTypeInfo):string;
var i:size_t;
  tensorInfo:TORTTensorTypeAndShapeInfo;
  sequenceInfo:TORTSequenceTypeInfo;
  mapInfo:TORTMapTypeInfo;
  strs:TStringArray;ints:TArray<int64_t>;
begin
  result:='';
  case _type.GetOnnxType() of
    ONNX_TYPE_UNKNOWN:
      result:='Unknowen';
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
      result:=result+ORTTypeInfoDesc(sequenceInfo.GetSequenceElementType())+')';
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



procedure TForm1.ComboBox1Change(Sender: TObject);
var
  InputNames,OutputNames:TArray<string>;i:Size_t;
  Info          :TOrtTypeInfo;
  MetaData      :TORTModelMetadata;
  version       :Int64_t;
  memInfo       :TORTMemoryInfo;
  s             :string;
  memtype       :OrtMemType;
  alloctype     :OrtAllocatorType;
  deviceid      :size_t;

begin
  BitBtn1.Enabled:=ComboBox1.ItemIndex>0;
  if ComboBox1.ItemIndex=0 then exit;
  ModelPath:=ComboBox1.Text;//'FasterRCNN-10.onnx'{'.onnx'};
  WriteLog('------------------ Start ----------------');
  WriteLog(format('Loading Model [%s]...',[ComboBox1.Text]));
  session:=TORTSession.Create(Modelpath);
  WriteLog('Model Loaded.');
  setLength(InputNames,Session.GetInputCount());
  for i:=0 to High(InputNames) do begin
    Info:=Session.GetInputTypeInfo(i);
    InputNames[i]:=string(session.GetInputNameAllocated(i,DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif})+' :'+ORTTypeInfoDesc(Info);
  end;
  setLength(OutputNames,Session.GetOutputCount());
  for i:=0 to High(OutputNames) do begin
    Info:=Session.GetOutputTypeInfo(i);
    OutputNames[i]:=string(session.GetOutputNameAllocated(i,DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif})+' :'+ORTTypeInfoDesc(Info);
  end;

  MetaData:=session.GetModelMetadata;
  WriteLog('Description : '+string(Metadata.GetDescriptionAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif}));
  WriteLog('Producer : '   +string(MetaData.GetProducerNameAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif}));
  WriteLog('Graph Name : ' +string(MetaData.GetGraphNameAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.instance{$endif}));
  WriteLog('Graph Description : '+string(MetaData.GetGraphDescriptionAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.instance{$endif}));
  WriteLog('Domain : '+string(MetaData.GetDomainAllocated(DefaultAllocator){$ifndef NO_SMARTPTR}.instance{$endif}));
  //version:=MetaData.GetVersion();
  //WriteLog('Version : '+FloatToStr(Version));
  WriteLog(format(' Inputs => Outputs : [%s] => [%s]',[Join(InputNames), Join(OutputNames)]));
  WriteLog('------------------- End ----------------');
  MemInfo:=AllocatorGetMemoryInfo(DefaultAllocator);
  s:=MemInfo.GetAllocatorName();
  WriteLog('Memtype :'      +IntToStr(Ord(MemInfo.GetMemoryType())   ));
  WriteLog('AllocatorType :'+IntToStr(Ord(MemInfo.GetAllocatorType())));
  WriteLog('Device Type : '+s);
  WriteLog('DeviceId :'     +IntToStr(Ord(MemInfo.GetDeviceId())     ));
end;

procedure TForm1.BitBtn1Click(Sender: TObject);
//var i,j:integer;

var Ratio:single; bmp:TBitmap;
    t1, paddedWidth, paddedHeight:size_t;
    y,x:int64;
    inSize, OutSize:TSize;
    input,boxes,confidence:TOrtTensor<single>;
    labels:TOrtTensor<int64>;
    outTensors:TArray<TORTValue>;
    mean:array of single =// [ 0,0,0];
                         [ 102.9801, 115.9465, 122.7717] ;
    pixelSpan:PBGRA; inputs,OutPuts:TORTNameValueList;


begin

  // only during debug on lazarus,we have to use <gdb> instead of FPDebug,
  // from lazarus IDR search for :
  // <DisableLoadSymbolsForLibraries> inside "Tools->Options->Debugger" and set it to <True>
  // otherwise the open dialog might freeze during a debug session
  if OpenDlg.Execute then Image1.Picture.LoadFromFile(OpenDlg.FileName) else exit;
  if Image1.Picture.Bitmap.PixelFormat<>pf32bit then
    raise Exception.create('pixel format is not 32bit');
  bmp:=TBitmap.Create;
  bmp.PixelFormat:=Image1.Picture.Bitmap.PixelFormat;
  with Image1.Picture do begin
    inSize.cx:=Bitmap.width;inSize.cy:=Bitmap.Height;
  end;
  ratio := 800.0 / Min(Image1.Picture.Bitmap.Width, Image1.Picture.Bitmap.Height);
  bmp.SetSize(
    trunc(Image1.Picture.Bitmap.width*ratio),
    trunc(Image1.Picture.Bitmap.Height*Ratio));
  outSize.cx:=bmp.width; outSize.cy:=bmp.height;
  //bmp.BeginUpdate(true);
  //bmp.Canvas.pixels[1,1]:=clblack;
  //bmp.EndUpdate();

  t1:=GetTickCount64;

  //bmp.Canvas.CopyRect(bmp.Canvas.ClipRect,Image1.Picture.Bitmap.Canvas, Image1.Picture.Bitmap.Canvas.ClipRect);
  //bmp.Canvas.StretchDraw(rect(0,0,bmp.Width, bmp.Height), Image1.Picture.Bitmap);
  ScaleImage32<TBitmap>(inSize, outSize, Image1.Picture.bitmap, bmp);

  t1:=GetTickCount64-t1;
  WriteLog(format('took [%d]ms : Original (%dx%d), Resized (%d,%d)',[t1, Image1.Picture.Bitmap.Width,Image1.Picture.Bitmap.Height, trunc(ratio *Image1.Picture.Bitmap.Width), trunc(ratio*Image1.Picture.Bitmap.Height)]));
  Application.ProcessMessages;

  t1:=GetTickCount64;
  paddedWidth  := Ceil(bmp.width  / 32) * 32;
  paddedHeight := Ceil(bmp.Height / 32) * 32;
  Input:=TOrtTensor<single>.Create([paddedHeight,paddedWidth,3 ]);
  for  y:= paddedHeight - bmp.Height to bmp.Height-1 do begin
    pixelSpan := bmp.scanline[y];
    for x := paddedWidth - bmp.Width to bmp.Width-1 do begin
        input[x,y,0] := pixelSpan[x].B - mean[0];
        input[x,y,1] := pixelSpan[x].G - mean[1];
        input[x,y,2] := pixelSpan[x].R - mean[2];
    end
  end;
  //Inputs:=TORTNameValueList.Create;
  inputs.AddOrSetValue('image',input.ToValue);

  // *********** here we go!!!!*************** \

  Outputs:=session.Run(Inputs) ;

  // *****************************************
  t1:=GetTickCount64-t1;
  Memo1.Lines.Add(format('Inferance took [%d]ms',[t1]));
  //TValueHelper.At();
  //for x:=0 to Outputs.Count-1 do
  //  begin
  //    WriteLog(Outputs.keys[x]+' => '+TensorTypes[Outputs.Values[x].GetTensorType()]+'['+join(Outputs.Values[x].GetTensorShape)+']');
  //  end;

  // the model will return boxes boxes at position 0;
  outTensors := Outputs.Values;
  boxes     := outTensors[0];
  labels    := outTensors[1];
  confidence:= outTensors[2];
  writelog('Boxes : '+join(boxes.shape)+' '+boxes.ToString) ;
  writelog('Confidence : '+join(confidence.shape)+' '+confidence.ToString) ;
  bmp.BeginUpdate(true);
  //bmp.canvas.pen.Style:=TPenStyle.psSolid;
  //bmp.Canvas.copymode:=cmSrcCopy;
  //bmp.PixelFormat:=pf32bit;
  bmp.Canvas.brush.Style:=TBrushstyle.bsClear;
  bmp.canvas.pen.Color:=clLime;
  bmp.canvas.pen.Width:=4;
  bmp.Canvas.Font.Size:=8;
  bmp.canvas.font.Color:=clWhite;
  //bmp.Canvas.rectangle(10,10,200,200);

  y:=0;
  for x:=0 to Boxes.Shape[1]-1 do
    if confidence.FData[x]> 0.5 then
    with  bmp.Canvas do begin
      bmp.canvas.Rectangle(round( boxes.index2[0,x]), round( boxes.index2[1,x]), round( boxes.index2[2,x]), round( boxes.index2[3,x]));
      bmp.Canvas.TextOut(round(boxes.index2[0,x])+3,round(boxes.index2[1,x])+3,_labels[labels.index1[x]]);
    end;
  bmp.EndUpdate();
  Image2.Picture.Graphic:=(bmp);
  freeAndNil(bmp);
end;

procedure TForm1.FormShow(Sender: TObject);
var sr:TSearchRec;
begin
  WriteLog('Providers : ['+ Join(GetAvailableProviders())+']');
    WriteLog('---------------------------------------');
  ComboBox1.Items.text:='-- Empty --';
  ComboBox1.ItemIndex:=0;
  if FindFirst('*.onnx',faNormal,sr)=0 then begin
    ComboBox1.Items.add(sr.Name);
    while FindNext(sr)=0 do
      ComboBox1.Items.add(sr.Name);
  end;
  FindClose(sr);
end;

procedure TForm1.WriteLog(const str:string);
begin
  memo1.lines.Add(str);
end;

procedure TForm1.BitBtn2Click(Sender: TObject);
  var
  inputs,OutPuts:TORTNameValueList;
  Inputx_s,inputy_s,output_s:TOrtTensor<single>;
  i:int64_t;
begin
    ModelPath:='model\'+ComboBox1.Text;
    Inputx_s:=TOrtTensor<single>.Create([10]);
    Inputy_s:=TOrtTensor<single>.Create([10]);
    for i:=0 to Inputx_s.Shape[0]-1 do
      begin
       Inputx_s.index1[i]:=(i+1)*0.2;
       Inputy_S.index1[i]:=(i+1);
      end;
    inputs['X']:=inputx_s;
    inputs['Y']:=inputy_S;
    Outputs:=session.Run(Inputs);
    output_s:= Outputs.Values[0];
    writelog('result : '+join(output_s.shape)+' '+output_s.ToString) ;

    for i:=0 to Inputx_s.Shape[0]-1 do
      begin
       Inputx_s.index1[i]:=(i+1)*2;
       Inputy_S.index1[i]:=(i+1);
      end;
    inputs['X']:=inputx_s;
    inputs['Y']:=inputy_S;
    Outputs:=session.Run(Inputs);
    output_s:= Outputs.Values[0];
    writelog('result : '+join(output_s.shape)+' '+output_s.ToString) ;

end;

end.

