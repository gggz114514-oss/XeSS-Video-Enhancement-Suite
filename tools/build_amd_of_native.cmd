@echo off
rem Public R4 source does not contain GPU Block / GPU DIS implementations.
echo [build] Native GPU workers require private sources and are not rebuilt here.
echo [build] Use the EXEs and shaders from the matching published R4 runtime.
echo [build] AMD provider sources remain public; its worker uses the same private native host.
exit /b 2
