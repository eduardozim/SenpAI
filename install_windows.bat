@echo off
setlocal
echo ==========================================
echo Script de Instalacao do SenpAI para Windows
echo ==========================================
echo.
echo Verificando Python 3.11...
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [AVISO] Python 3.11 (via py launcher) nao foi encontrado.
    echo Certifique-se de instalar a versao oficial do Python 3.11.
    echo Voce pode rodar o comando: winget install Python.Python.3.11
    pause
    exit /b 1
)

echo Criando o ambiente virtual (.venv)...
py -3.11 -m venv .venv
if %errorlevel% neq 0 (
    echo [ERRO] Falha ao criar o ambiente virtual.
    pause
    exit /b 1
)

echo Ativando o ambiente virtual e instalando as bibliotecas...
call .\.venv\Scripts\activate.bat

echo Atualizando pip...
python -m pip install --upgrade pip

echo Instalando requerimentos base...
pip install -r requirements.txt

echo Verificando presenca de placa NVIDIA...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo Placa NVIDIA detectada. Instalando aceleracao CUDA...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install ultralytics onnxruntime-gpu
) else (
    echo Nenhuma placa NVIDIA detectada ou nvidia-smi nao esta no PATH. Pulando CUDA.
)

echo.
echo ==========================================
echo Instalacao concluida com sucesso!
echo ==========================================
echo.
echo Para iniciar o SenpAI, execute os comandos abaixo:
echo.
echo   .\.venv\Scripts\activate.bat
echo   python -m streamlit run app.py
echo.
pause
