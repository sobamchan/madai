import typer

from madai.chi2 import chi2
from madai.spearman import spearman

app = typer.Typer()
app.command()(chi2)
app.command()(spearman)
