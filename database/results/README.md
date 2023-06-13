# Analysis results

This directory contains the results of the analysis.
All results, whether stored in text files (e.g. `.txt` files) or binary files (e.g. `.root` files), will be encrypted before being committed by git.

Encrypted files are always named with the `.enc` extension.
For example, `results.txt` will be encrypted as `results.txt.enc`. Vice versa, files without the `.enc` extension are assumed to be unencrypted, and they are ignored by git.
For example, `results.txt` will not and must not be committed to git.

## Usage

You need to set up the encryption key before you can use any of these files. Please contact the maintainers for more details.

First, you need to activate the conda environment, and make sure it has been set up correctly.
Symbolic links at `$CONDA_PREFIX/bin` to the scripts in [`$PROJECT_DIR/local/bin`](../../local/bin) should have been created when you installed `env_e15190`.
We will be using the command [`encryption`](../../local/bin/encryption).

Once you have everything set up, you can decrypt a file as follows:
```console
(env_e15190) user@server $ encryption result.txt.enc
Decrypted "result.txt.enc" to "result.txt".
```
You can now interact with the file `result.txt` as you would normally do.

If you have modified the result or created new result, you can encrypt it as follows:
```console
(env_e15190) user@server $ encryption result.txt
Encrypted "result.txt" to "result.txt.enc".
```
You can now commit the encrypted `result.txt.enc` to git.

To see more options, run `encryption --help`.

## Security

Please note that the encryption related files are stored as `.gcm_key` and `.gcm_salt` in the root directory of this repository.
Both files are ignored by git, but it is still important not to accidentally force git to track them.
Also, please do not share these files with anyone else who is not authorized to access the encrypted files.
