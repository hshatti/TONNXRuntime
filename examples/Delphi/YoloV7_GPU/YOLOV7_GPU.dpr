program YOLOV7_GPU;

uses
  System.StartUpCopy,
  FMX.Forms,  ComObj,
  sysUtils,
  uMainForm in 'uMainForm.pas' {Form1},
  uScreenForm in 'uScreenForm.pas' {ScreenForm},
  onnxruntime.dml in '..\..\..\source\onnxruntime.dml.pas';

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.CreateForm(TScreenForm, ScreenForm);
  Application.Run;
end.
