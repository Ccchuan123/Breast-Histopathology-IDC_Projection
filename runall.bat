@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo   Breast Histopathology IDC - Auto Run
echo ========================================
echo.

set PYTHONIOENCODING=utf-8

echo [1/4] Stage 1: Traditional ML Baseline ...
python experiments/stage1_baseline_ml.py
if %errorlevel% neq 0 (
    echo FAILED: stage1_baseline_ml.py
    pause
    exit /b 1
)
echo OK
echo.

echo [2/4] Stage 2: ResNet18 Frozen Backbone ...
python experiments/stage2_resnet_frozen.py
if %errorlevel% neq 0 (
    echo FAILED: stage2_resnet_frozen.py
    pause
    exit /b 1
)
echo OK
echo.

echo [3/4] Stage 3: ResNet18 Fine-tuning ...
python experiments/stage3_resnet_finetune.py
if %errorlevel% neq 0 (
    echo FAILED: stage3_resnet_finetune.py
    pause
    exit /b 1
)
echo OK
echo.

echo [4/4] Stage 4: Model Comparison ^& Ensemble ...
python experiments/stage4_model_comparison.py
if %errorlevel% neq 0 (
    echo FAILED: stage4_model_comparison.py
    pause
    exit /b 1
)
echo OK
echo.

echo ========================================
echo ALL DONE!
echo Models:   outputs\models\
echo Figures:  outputs\figures\
echo Results:  outputs\all_model_results.csv
echo ========================================
pause
