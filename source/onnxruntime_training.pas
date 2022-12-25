unit onnxruntime_training;

{$ifdef fpc}
  {$PACKRECORDS C}
  {$mode DELPHI}
  {$ModeSwitch advancedrecords}
  {$ModeSwitch typehelpers}
  {$define outvar:=var}
  {$PointerMath on}
{$else}
  {$define outvar:=out}
{$endif}

{$define ORT_API_CALL:=stdcall}
{$H+}

interface

uses
  SysUtils;

implementation

end.

