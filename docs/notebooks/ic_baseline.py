import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from yukka.data import Index  # the universe, for its RICs

    from novetly_score.data import data  # the entry point

    return Index, data


@app.cell
def _(Index, data):
    frame = data(Index.STOXX600.frame).counts(key="ric")
    return (frame,)


@app.cell
def _(frame):
    frame
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
