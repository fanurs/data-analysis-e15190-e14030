# How to update documentation?

Make sure you have activated the conda environment. Then enter `$PROJECT_DIR/docs` and run `refresh.sh`:
```console
cd $PROJECT_DIR/docs
./refresh.sh
```

Sphinx will then auto-generate the HTML documentation at `$PROJECT_DIR/docs/build/html/`. Inspect the `*html` files locally using your browser before committing them to GitHub. Once uploaded, the documentation will be statically at https://fanurs.github.io/data-analysis-e15190-e14030/.