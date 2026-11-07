# BrawlIA V2 — Fase 1 com Windows EXE e OTA GitHub

Projeto V2 **independente da V1.x**: mede captura e entrega de quadros, sem controle de partidas, OCR, gravador de PNGs ou vídeo. 30 FPS é **meta de captura/consumo**, não taxa garantida.

## Usar no Windows (sem Python)

1. Baixe **BrawlIA-V2-Windows.zip** na [release mais recente](https://github.com/K410Sh4/BrawlIa/releases/latest).
2. Extraia **BrawlIA-V2.exe** e **BrawlIA-V2-Updater.exe** na mesma pasta de usuário (ex.: `Downloads\BrawlIA-V2`). Não execute de dentro do ZIP.
3. Dê duplo clique em **BrawlIA-V2.exe**. Não é necessário abrir `.bat` ou instalar Python.
4. Abra o BlueStacks visível, com resolução **1280×720 e 30 FPS**, e clique em **Iniciar teste 30 FPS**. O botão também encerra e grava as métricas.
5. Clique em **Abrir relatórios**. O caminho é `%LOCALAPPDATA%\BrawlIA-V2\relatorios`.

Ao iniciar, o programa consulta o GitHub Releases para verificar novas versões. Quando houver uma versão superior, oferece a instalação **com confirmação**. O arquivo é baixado e comparado ao SHA-256 publicado no manifesto da mesma release. O atualizador separado espera o programa fechar, substitui-o e mantém um `.previous.exe` para recuperação. **Integridade do arquivo não equivale a assinatura de código**; distribua apenas via releases do repositório oficial.

Se o GitHub estiver indisponível, a captura continua funcionando. O programa não envia dados da partida ao GitHub.

## Criação do `.exe` e publicação

O workflow `.github/workflows/brawlia-v2-release.yml` é executado no GitHub Actions (Windows). Executa os testes, compila dois executáveis com PyInstaller, cria pacote ZIP, publica um artefato de execução e publica GitHub Release estável quando o token tem `contents: write`. Para novos lançamentos, aumente `brawlia_v2/__init__.py:__version__`, por exemplo de `2.0.0` para `2.0.1`, e publique novamente. Releases não são sobrescritas silenciosamente.

### Rodar via Python (alternativa)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_gui.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

A falha anterior do `.bat` foi tratada com inicializador de fallback que verifica a presença de Python e retorna mensagens claras. O executável dispensa esse fallback.

## Limitações

- A Fase 1 mede somente captura e integridade dos quadros, ainda não implementa a IA da Fase 2.
- A compilação **Windows** precisa de um runner Windows real. Não se deve apresentar um EXE Linux renomeado como Windows.
- Se sua instalação estiver em `Program Files` sem permissão de gravação, a atualização pode falhar. Use uma pasta pessoal.
- Antivírus/SmartScreen podem solicitar confirmação para um novo binário não assinado. A publicação do GitHub não assina código Windows.
- Utilização de CPU/GPU não está sendo medida diretamente; somente tempos e FPS efetivos.
