@echo off
REM compile
javac -cp "lib/*" -d bin src\Main.java
IF ERRORLEVEL 1 (
  echo Compile failed.
  pause
  exit /b 1
)

REM run with module opens to allow Weka to inject MTJ/native classes
java --add-opens java.base/java.lang=ALL-UNNAMED ^
     --add-opens java.base/java.lang.reflect=ALL-UNNAMED ^
     -cp "bin;lib/*" src.Main data\Companies.csv

pause
