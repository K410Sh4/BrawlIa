"""python report_session.py gravacoes/SESSAO/analise.jsonl --output relatorio.json"""

import argparse
from pathlib import Path
from brawlla.metrics import summarize_log
from brawlla.reporting import save_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relatório de uma sessão BrawlIA (V0.3 ou posterior)."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("relatorio.json"))
    args = parser.parse_args()
    try:
        save_report(summarize_log(args.log), args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Erro: {exc}\n")
    print(
        f'Relatórios: {args.output.resolve()} e {args.output.with_suffix(".html").resolve()}'
    )
