program FasterRCNN;    {$APPTYPE Console}

uses
  System.StartUpCopy,
  FMX.Forms,
  delphi_unit in 'delphi_unit.pas' {Form1};

{$R *.res}

begin
  ReportMemoryLeaksOnShutdown:=true;
  Application.Initialize;
  Application.CreateForm(TForm1, Form1);
  Application.Run;
end.
