# F1 ML Lab - update

Replace / add these files in your project (everything else - style.css, extra.css, charts.css, data/ - stays as it was):

    app.py                      replaced
    ml/__init__.py              new (empty)
    ml/common.py                new  (preprocessor + constants, moved out of pipeline.py)
    ml/advanced.py              new  (methods from the literature review)
    ml/pipeline.py              replaced
    templates/index.html        replaced
    static/js/app.js            replaced
    static/css/app.css          new

Run:  python app.py   ->  http://127.0.0.1:5000
Open "Run Settings", pick models (or "FAST SET ONLY"), press RUN. Progress is live.
Results are saved to results/latest.json and results/bundle.joblib, so the site shows the
previous run immediately after a restart.

Command line (no website):  python ml/pipeline.py data plots --quick
