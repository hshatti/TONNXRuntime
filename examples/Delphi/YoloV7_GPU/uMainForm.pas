unit uMainForm;

interface

uses
  Windows,System.SysUtils, System.Types, System.UITypes, System.Classes, System.Variants,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs,
  FMX.Controls.Presentation, FMX.StdCtrls, FMX.Layouts, FMX.ExtCtrls,
  FMX.Objects, uScreenForm,psAPI, FMX.Platform    ;

type
  TForm1 = class(TForm)
    Button1: TButton;
    Button2: TButton;
    Image1: TImage;
    StatusLabel: TLabel;
    RAMLabel: TLabel;
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Image1Paint(Sender: TObject; Canvas: TCanvas;
      const ARect: TRectF);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormShow(Sender: TObject);
  private
    bmp:TBitmap;
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation

{$R *.fmx}
procedure TForm1.Button1Click(Sender: TObject);

begin
  DetReset;
  if FileExists(GetCurrentDir+'..\..\startrek1.jpg') then
    bmp.LoadFromFile(GetCurrentDir+'..\..\startrek1.jpg')
  else
    bmp.LoadFromFile('startrek1.jpg');

  Image1.Bitmap.Assign(bmp);
end;

function CurrentProcessMemory: Cardinal;
var
  MemCounters: TProcessMemoryCounters;
begin
  MemCounters.cb := SizeOf(MemCounters);
  if GetProcessMemoryInfo(GetCurrentProcess,
      @MemCounters,
      SizeOf(MemCounters)) then
    Result := MemCounters.WorkingSetSize
  else
    RaiseLastOSError;
end;

procedure TForm1.Button2Click(Sender: TObject);
var
  h,w: Integer;
  i: Integer;
begin
  if Image1.Bitmap.IsEmpty then
    begin
      MessageDlg('Load picture first!',TMsgDlgType.mtError,[TMsgDlgBtn.mbOK],0);exit
    end;
  w := Bmp.Canvas.Width;
  h := Bmp.Canvas.Height;
  try
    i := InferenceFromBitmap(bmp, enable_nms);
  except
    i :=-108;
  end;
  if i <0 then
  begin
    StatusLabel.Text:='Inference failed Code' + IntToStr(i);
  end else begin
    Image1.Bitmap:=Bmp;
    Application.ProcessMessages; //ensure image is assigned
    Image1.Repaint;
    StatusLabel.Text:=Format(
      'Width: %d'+sLineBreak+
      'Height: %d'+sLineBreak+
      'ThrsholdCount: %d'+sLineBreak+
      'Detections:%d'+sLineBreak+
      'InferenceTime: %u(ms)'+sLineBreak+
      'nmsTime: %u'+sLineBreak+
//      'nmsEnabled: %g' +
      'ExitCode: %d'
      ,[w,h,thresholded_cnt,det_cnt,det_inference_time,det_nms_time{,enable_nms},i]);
    Button2.Text:='Start Inference';
    RamLabel.Text:=IntToStr(Round(CurrentProcessMemory/1024.0/1024.0)) + ' MB';

  end;
end;


procedure TForm1.FormCreate(Sender: TObject);
begin
  Bmp:=TBitmap.Create;
end;

procedure TForm1.FormDestroy(Sender: TObject);
begin
  FreeAndNil(bmp)
end;


procedure TForm1.FormShow(Sender: TObject);
begin
  ScreenForm.Show
end;

const squreColors : array[0..10] of TAlphaColor=(
  $FF800000, $FF008000, $FF000080, $FF808000, $FF008080, $FF800080, $FF000040, $FF008040, $FF0080C0,$FF408000,$FFC080C0
 );
procedure TForm1.Image1Paint(Sender: TObject; Canvas: TCanvas;
  const ARect: TRectF);
var
  ZRect: TRectF;
  XRadius, YRadius: Single;
  x: Integer;
  x1, y1, x2, y2: Single;
  cls: Integer;
  score: Single;
  area: Single;

begin
  if Image1.Bitmap.IsEmpty then
     Exit();
  if det_running then
     Exit();

  with Image1.Bitmap.Canvas do
  begin
    BeginScene;
    for x := 0 to det_cnt-1 do
      begin
        x1 := det_boxes[x, 0];
        y1 := det_boxes[x, 1];
        x2 := det_boxes[x, 2];
        y2 := det_boxes[x, 3];
        cls := det_classes[x];
        score := det_argmax_score[x];
        area := det_areas[x];
        ZRect := TRectF.Create(x1, y1, x2, y2);
        XRadius := 5;
        YRadius := 5;
        if cls <11 then
          Stroke.Color := squreColors[cls]
        else
          Stroke.Color := TAlphaColor($FF808080);
        // draw bounding box
        Stroke.Thickness := 5.0;
        DrawRect(ZRect, XRadius, YRadius, AllCorners, 1, TCornerType.Round);
        // fill title bounding box with 50% transparent black
        Stroke.Thickness := 0.8;
        Font.Size := 24;
        Font.Family := 'Roboto';
        Font.Style := [TFontStyle.fsbold];
        ZRect := TRectF.Create(x1 + 3, y1 + 3, x2 - 3, y1 + Font.Size + 6);
        // draw background layer text in pure black
        Fill.Color:=TAlphaColor($80000000);
        FillRect(ZRect, XRadius, YRadius, AllCorners, 1, TCornerType.Round);
        Fill.Color := TAlphaColorRec.Silver;
        FillText(ZRect, format(' %d::%.3f',[cls,score]), false, 1, [], TTextAlign.Leading);
      end;
    EndScene;
  end;
end;

end.
