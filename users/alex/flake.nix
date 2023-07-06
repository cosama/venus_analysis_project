{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs";
  outputs = {
    self,
    nixpkgs,
  }: {
    devShells.x86_64-linux.default = with import nixpkgs {
      system = "x86_64-linux";
    };
      mkShell {
        buildInputs = [
          pkgs.python3Full
          pkgs.python310Packages.matplotlib
          pkgs.python310Packages.numpy
          pkgs.python310Packages.pandas
          pkgs.python310Packages.scikit-learn
          pkgs.python310Packages.tqdm
          pkgs.python310Packages.xgboost
          pkgs.python310Packages.fastparquet

          pkgs.gnome.eog # view images
        ];

        shellHook = ''
          ${pkgs.zsh}/bin/zsh
          exit'';
      };
  };
}
