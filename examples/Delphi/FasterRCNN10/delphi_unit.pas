unit delphi_unit;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes, System.Variants,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs, Generics.Collections,
  FMX.StdCtrls, FMX.ListBox, FMX.Controls.Presentation, FMX.Memo.Types, math,
  FMX.ScrollBox, FMX.Memo, FMX.Objects,onnxruntime_pas_api,onnxruntime;

  type
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

  TForm1 = class(TForm)
    Panel1: TPanel;
    ComboBox1: TComboBox;
    Label1: TLabel;
    Button1: TButton;
    CheckBox1: TCheckBox;
    Button2: TButton;
    Memo1: TMemo;
    Panel2: TPanel;
    Image1: TImage;
    Splitter1: TSplitter;
    Image2: TImage;
    OpenDlg: TOpenDialog;
    Splitter2: TSplitter;
    Label2: TLabel;
    procedure FormShow(Sender: TObject);
    procedure ComboBox1Change(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure CheckBox1Change(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var  Form1: TForm1;
  ModelPath:string ;
  session:TORTSession;
  input:TORTValue;

function Join(const arr:array of ansistring; const Delimiter:ansistring=', '):ansistring;  overload;
function Join(const arr:array of int64_t; const Delimiter:ansistring=', '):ansistring;  overload;
function Join(const arr:array of double; const Delimiter:ansistring=', '):ansistring;  overload;
function TypeInfoDesc(const _type:TOrtTypeInfo):ansistring;
procedure  WriteLog(str:ansistring);

implementation

{$R *.fmx}


//procedure ScaleImage32(const inSize,outSize:TSize;const inData, outData:TBitmap;scaleType:TScaleType=stNearest);
//var i,j,i1,j2:integer;p1,p2:TBGRA; pDst,pSrc,pSrc2:PBGRA; tx,ty:single;
//function lerp(const a,b:TBGRA;const t:single):TBGRA; inline;
//
//begin
//
//  result.v[0]:=round(a.v[0]+(b.v[0]-a.v[0])*t);
//  result.v[1]:=round(a.v[1]+(b.v[1]-a.v[1])*t);
//  result.v[2]:=round(a.v[2]+(b.v[2]-a.v[2])*t);
//  //result.v[3]:=round(a.v[3]+(b.v[3]-a.v[3])*t);
//end;
//
//var xRatio,yRatio:single;
//begin
//  xRatio:=inSize.cx/outSize.cx;
//  yRatio:=inSize.cy/outSize.cy;
//  if scaleType=stNearest then begin
//    for i:=0 to outsize.cy-1 do begin
//      pDst:=outData.scanline[i];
//      pSrc:=inData.ScanLine[round(i*yRatio)];
//      for j:=0 to outsize.cx-1 do begin
//        pDst^:=pSrc[round(j*xRatio)];
//        inc(pDst);
//        //outData.Canvas.pixels[j,i]:=inData.canvas.pixels[round(j*xRatio),round(i*yRatio)]
//      end;
//    end;
//    exit
//  end;
//  for i:=0 to outSize.cy-2 do begin
//    pDst:=outData.scanline[i];
//    pSrc:=inData.ScanLine[round(i*yRatio)];
//    pSrc2:=inData.ScanLine[round((i+1)*yRatio)];
//    ty:=frac(i*yRatio);
//    for j:=0 to outsize.cx-1 do begin
//      tx:=frac(j*xRatio);
//      p1:=lerp( pSrc[round(j*xRatio)],  pSrc[round((j+1)*xRatio)],tx);
//      p2:=lerp(pSrc2[round(j*xRatio)], pSrc2[round((j+1)*xRatio)],tx);
//      pDst^:=lerp(p1,p2,ty);
//      inc(pDst);
//    end;
//  end;
//  pSrc:=inData.ScanLine[round((outSize.cy-1)*yRatio)];
//  for j:=0 to outsize.cx-1 do
//    pDst[j]:=pSrc[round(j*xRatio)];
//
//end;

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

procedure  WriteLog(str:ansistring);
begin
  form1.memo1.lines.add(str)
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

var
  mean:array of single =// [ 0,0,0];
                         [ 102.9801, 115.9465, 122.7717] ;
procedure TForm1.Button1Click(Sender: TObject);
var Ratio:single; bmp:TBitmap;
    t1, paddedWidth, paddedHeight:size_t;
    y,x:int64;
    inSize, OutSize:TSize;
    input,boxes,confidence:TOrtTensor<single>;
    labels:TOrtTensor<int64>;
    TensorDim:TArray<int64_t>;

    pixelSpan:TBitmapData;
    inputs,Outputs:TORTNameValueList;


begin

  // only during debug on lazarus, use <gdb> instead of FPDebug, search for :
  // <DisableLoadSymbolsForLibraries> inside "Tools->Options->Debugger" and set it to <True>
  // otherwise the open dialog might freeze during a debug session
  if OpenDlg.Execute then Image1.Bitmap.LoadFromFile(OpenDlg.FileName) else exit;
  Application.ProcessMessages;
  if Image1.Bitmap.PixelFormat<>TPixelFormat.BGRA then
    raise Exception.create('pixel format is not 32bit');
  bmp:=TBitmap.Create;
  with Image1 do begin
    inSize.cx:=Bitmap.width;inSize.cy:=Bitmap.Height;
  end;
  ratio := 800.0 / Min(Image1.Bitmap.Width, Image1.Bitmap.Height);
  bmp.SetSize(
    trunc(Image1.Bitmap.width*ratio),
    trunc(Image1.Bitmap.Height*Ratio));
  outSize.cx:=bmp.width; outSize.cy:=bmp.height;
  t1:=TThread.GetTickCount;
  bmp.Canvas.BeginScene();
  bmp.Canvas.DrawBitmap(Image1.Bitmap,rectf(0,0,image1.Bitmap.Width, image1.Bitmap.Height),rectf(0,0,bmp.width,bmp.height),255);
  bmp.Canvas.EndScene;
//  ScaleImage32<TBitmap>(inSize, outSize, Image1.Picture.bitmap, bmp);
  t1:=TThread.GetTickCount-t1;
  Memo1.Lines.Add(format('took [%d]ms : Original (%dx%d), Resized (%d,%d)',[t1, Image1.Bitmap.Width,Image1.Bitmap.Height, trunc(ratio *Image1.Bitmap.Width), trunc(ratio*Image1.Bitmap.Height)]));
  t1:=TThread.GetTickCount;
  paddedWidth  := Ceil(bmp.width  / 32) * 32;
  paddedHeight := Ceil(bmp.Height / 32) * 32;
  TensorDim := [3, paddedWidth, paddedHeight];
  Input:=TOrtTensor<single>.Create([paddedHeight,paddedWidth,3 ]);
  bmp.Map(TMapAccess.Read,pixelSpan);
  for  y:= paddedHeight - bmp.Height to bmp.Height-1 do begin
    for x := paddedWidth - bmp.Width to bmp.Width-1 do begin
        ;
        input[x,y,0] := TAlphaColorRec(pixelSpan.GetPixel(x,y)).B - mean[0];
        input[x,y,1] := TAlphaColorRec(pixelSpan.GetPixel(x,y)).G - mean[1];
        input[x,y,2] := TAlphaColorRec(pixelSpan.GetPixel(x,y)).R - mean[2];
    end
  end;
  bmp.Unmap(pixelSpan);
//  Inputs:=TORTNameValueList.Create;
  inputs.AddOrSetValue('image',input.ToValue);

  Outputs:=session.Run(Inputs) ;
  t1:=TThread.GetTickCount-t1;
  Memo1.Lines.Add(format('Inferance took [%d]ms',[t1]));
  //TValueHelper.At();
  //for x:=0 to Outputs.Count-1 do
  //  begin
  //    WriteLog(Outputs.keys[x]+' => '+TensorTypes[Outputs.Values[x].GetTensorType()]+'['+join(Outputs.Values[x].GetTensorShape)+']');
  //  end;

  // the model will return boxes boxes at position 0;
  boxes     := Outputs.Values[0];
  confidence:= Outputs.Values[2];
  writelog('Boxes : '+join(boxes.shape){+' '+boxes.ToString}) ;
  writelog('Confidence : '+join(confidence.shape){+' '+confidence.ToString}) ;
  //bmp.canvas.pen.Style:=TPenStyle.psSolid;
  //bmp.Canvas.copymode:=cmSrcCopy;
  //bmp.PixelFormat:=pf32bit;
  bmp.Canvas.BeginScene();
  bmp.Canvas.Fill.Kind:=TBrushKind.None;
  bmp.canvas.Stroke.Color:=TAlphaColorRec.Lime;
  bmp.canvas.Stroke.Thickness:=4;
  bmp.canvas.Stroke.Kind:=TBrushKind.Solid;

  for x:=0 to Boxes.Shape[1]-1 do
    if confidence.index1[x]> 0.5 then
    with  bmp.Canvas do begin
      bmp.canvas.DrawRect(Rectf( boxes.index2[0,x], boxes.index2[1,x], boxes.index2[2,x], boxes.index2[3,x] ),0,0,[],255);
    end;
  bmp.Canvas.EndScene;

  Image2.Bitmap:=bmp;
  freeAndNil(bmp);
end;


procedure TForm1.CheckBox1Change(Sender: TObject);
begin
//  Image1.Stretch:=Checkbox1.Checked;
//  Image1.StretchInEnabled:=Checkbox1.Checked;
//  Image1.StretchOutEnabled:=Checkbox1.Checked;
//  Image1.AutoSize:=not Checkbox1.Checked;
//  Image1.SetBounds(0,0,ScrollBox1.Width,Scrollbox1.Height);
//  Scrollbox1.HorzScrollBar.Range:=0;
//  Scrollbox1.VertScrollBar.Range:=0;
//
//  Image2.Stretch:=Checkbox1.Checked;
//  Image2.StretchInEnabled:=Checkbox1.Checked;
//  Image2.StretchOutEnabled:=Checkbox1.Checked;
//  Image2.AutoSize:=not Checkbox1.Checked;
//  Image2.SetBounds(0,0,ScrollBox2.Width,Scrollbox2.Height);
//  Scrollbox2.HorzScrollBar.Range:=0;
//  Scrollbox2.VertScrollBar.Range:=0;
end;

procedure TForm1.ComboBox1Change(Sender: TObject);
var
  InputNames,OutputNames:TArray<ansistring>;i:Size_t;
  Info          :TORTTypeInfo;
  MetaData      :TORTModelMetadata;
  version       :Int64_t;
  memInfo       :TORTMemoryInfo;
  s             :ansistring;
  memtype       :OrtMemType;
  alloctype     :OrtAllocatorType;
  deviceid      :size_t;

begin
  Button1.Enabled:=ComboBox1.ItemIndex>0;
  if ComboBox1.ItemIndex=0 then exit;
  ModelPath:=ComboBox1.Items[ComboBox1.ItemIndex];//'FasterRCNN-10.onnx'{'.onnx'};
  WriteLog('------------------ Start ----------------');
  WriteLog(format('Loading Model [%s]...',[ModelPath]));
  session:=TORTSession.Create(Modelpath);
  WriteLog('Model Loaded.');
  setLength(InputNames,Session.GetInputCount());
  for i:=0 to High(InputNames) do begin
    Info:=Session.GetInputTypeInfo(i);
    InputNames[i]:=ansistring(session.GetInputNameAllocated(i,DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif})+' :'+TypeInfoDesc(Info);
  end;
  setLength(OutputNames,Session.GetOutputCount());
  for i:=0 to High(OutputNames) do begin
    Info:=Session.GetOutputTypeInfo(i);
    OutputNames[i]:=ansistring(session.GetOutputNameAllocated(i,DefaultAllocator){$ifndef NO_SMARTPTR}.Instance{$endif})+' :'+TypeInfoDesc(Info);
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
  MemInfo:=DefaultAllocator.GetMemoryInfo();
  s:=MemInfo.GetAllocatorName();
  WriteLog('Memtype :'      +IntToStr(Ord(MemInfo.GetMemoryType())   ));
  WriteLog('AllocatorType :'+IntToStr(Ord(MemInfo.GetAllocatorType())));
  WriteLog('Device Type : '+s);
  WriteLog('DeviceId :'     +IntToStr(Ord(MemInfo.GetDeviceId())     ));

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


end.
