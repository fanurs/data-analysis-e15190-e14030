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

# How to view remote HTML files using a local browser?

Unless you have installed this repository on your local machine, you will need some way to view HTML files remotely on your local browser. We can use something called "port forwarding".

1.
    Log on to the *remote* server (e.g. flagtail) and `cd` to the directory `$PROJECT_DIR/docs`. Make sure you have activated the conda environment. Then type the following command to start running a server on the remote machine:
    ```console
    python -m http.server 6666
    ```
    Here, `6666` is just an arbitrary one-time port number. You may choose whatever you like, as long as it has not been taken by other users.
    > :exclamation: For whatever reason, invoking `http.server` via the `-m` flag no longer works on Fishtank. If that is the case for you, try using the script at [here](https://gist.github.com/fanurs/2fb530d9f09bfa8dad8aa672f9d3d32c). As always, you can make this script an executable at, for example, `~/.local/bin`, for convenience.
    
    Occasionally, Python's `http.server` does not work on Fishtank. In that case, you can use Ruby to do the same thing:
    ```console
    ruby -run -ehttpd . -p6666
    ```
    Ruby is pre-installed on many Linux distros, including the one Fishtank is using.

1.
    Now, open up the terminal on your *local* computer. We will be "listening" to the *remote* port number from our *local* computer. On Linux or Mac, open up a terminal and type the following command:
    ```console
    ssh -J username@nsclgw.nscl.msu.edu -N -f -L localhost:6666:localhost:6666 username@flagtail
    ```
    If you are on a Windows machine, open up PowerShell and type the following command:
    ```powershell
    ssh.exe -J username@nsclgw.nscl.msu.edu -N -f -L localhost:6666:localhost:6666 username@flagtail
    ```
    The only difference here is Windows recognizes `ssh.exe` instead of `ssh`. Also, PowerShell will appear to look like it is frozen after you enter the passphrase - but it's not! Move on to next step.

    The command we show here is a little longer than what you might find on the Internet because the lab's Fishtank server has a "gateway" machine between the actual server (flagtail) and our computer. So we need to first "jump host" to the gateway (the `-J` part), then access flagtail. Adjust the other variables like `username` and `6666` accordingly.
    
    **Visual Studio Code**
    
    Some text editor or IDE (with plugins installed) also support port forwarding. If you have been using Visual Studio Code to connect to Fishtank via the [SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh), then you can just click on the "PORTS" tab on your panel, and input `6666`:
    <img width="400" alt="image" src="https://user-images.githubusercontent.com/21100851/156692846-845aaef0-8cd1-43b9-adf4-fa4005f8277c.png"><br>
    Once you are done, you will see something to the screenshot below. You can now proceed to the next step.<br>
    <img width="400" alt="image" src="https://user-images.githubusercontent.com/21100851/156692785-a61b2817-ca5c-4e0a-ad5c-8d1ac4cc8967.png"><br>
    In case the GUI is not the same on your side, simply trigger the Command Palette with "<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>" (Windows/Linux) or "<kbd>Command</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>" (Mac). Then search for "Forward a Port" and follow the instructions.<br>
    <img width="450" alt="image" src="https://user-images.githubusercontent.com/21100851/156693793-9ed78c47-a4c1-4b76-a8a5-9ddbcd643ee8.png">

1.
    Port forwarding should have been established. Just open up any web browser on your *local* computer, then input the following at the address bar:
    ```
    localhost:6666
    ```
    You should now see the documentation website. Congratulations! :fireworks:
