{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/languages/
  languages.rust = {
    enable = true;
    channel = "nightly";
  };

  # See full reference at https://devenv.sh/reference/options/
}
