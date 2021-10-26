# How to update documentation?

**The complete procedure**

Make sure you have activated the conda environment. Then in your linux terminal, enter `$PROJECT_DIR/docs` and run `refresh.sh`:
```console
cd $PROJECT_DIR/docs
./refresh.sh
```

Sphinx will then auto-generate the HTML documentation at `$PROJECT_DIR/docs/build/html/`. Inspect the `*.html` files locally using your browser before committing them to GitHub. Once uploaded, the documentation will be statically hosted at https://fanurs.github.io/data-analysis-e15190-e14030/.

**A faster way**

The `refresh.sh` script cleans up all previous built files, and recompile all the documentation from scratch. So it is usually a time-consuming process (~minutes). But when you are still actively writing the [docstrings](https://en.wikipedia.org/wiki/Docstring) for your code, you may want a faster way to update the modified `*.html` files, so that you can take a quick look at the preview of the website. In that situation, a faster command would be:
```console
cd $PROJECT_DIR/docs
make html
```

Of course, it is always recommended to run `./refresh.sh` at least one last time before deploying the files to GitHub. This is because sometimes the `make html` might still be using some old cache files that, for some unknown reason, were not cleaned up properly, changing the final appearance of the webpages.

# How to view remote HTML files using a local browser? (Linux/MacOS)

The following instructions only work on Linux or MacOS; I have yet to write down the instructions for Windows. :sweat_smile:

We need something called "port forwarding".

1. Log on to the *remote* server (e.g. flagtail) and `cd` to the directory `$PROJECT_DIR/docs`. Make sure you have activated the conda environment. Then type the following command to start running a server on the remote machine:
```console
python -m http.server 6666
```
Here, `6666` is just an arbitrary one-time port number. You may choose whatever you like, as long as it has not been taken by other users.

2. Now, open up the terminal on your *local* computer. We will be "listening" to the *remote* port number from our *local* computer. This can be done by typing:
```console
ssh -J username@nsclgw.nscl.msu.edu -N -f -L localhost:6666:localhost:6666 username@flagtail
```
This command is a little longer than what you might find on the Internet because the lab's Fishtank server has a "gateway" machine between the actual server (flagtail) and our computer. So we need to first "jump host" to the gateway (the `-J` part), then access flagtail. Adjust the other variables like `username` and `6666` accordingly.

3. Port forwarding should have been established. Just open up any web browser on your *local* computer, then input the following at the address bar:
```
localhost:6666
```
You should now be seeing the documentation website. Congratulations! :fireworks:

