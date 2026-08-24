# Building WebArena Images and Manual Setup

This replaces step 2 of [vm-setup.md](vm-setup.md) when you do not want to pull
prebuilt images from a container registry or download assets from a GCS bucket.
Run everything here on the `Websites VM`, then return to
[step 3](vm-setup.md#3-install-env_server-dependencies).

## Download the image tars and required artifacts

If you prefer building the WebArena images yourself, download the following image tars and required artifacts by following the instructions in [https://github.com/gasse/webarena-setup/tree/main/webarena](https://github.com/gasse/webarena-setup/tree/main/webarena):
- shopping_final_0712.tar
- shopping_admin_final_0719.tar
- postmill-populated-exposed-withimg.tar
- gitlab-populated-final-port8023.tar
- wikipedia_en_all_maxi_2022-05.zim
- openstreetmap-website-db.tar.gz
- openstreetmap-website-web.tar.gz
- openstreetmap-website.tar.gz

and place them in [../../environment_setup/webarena/archive](../../environment_setup/webarena/archive)

Then, run
```sh
cd ~/amber
bash environment_setup/webarena/build_images.sh
```

## Manual setup
If you don't want to upload images to an artifact registry, first make sure the images are created on the website server VM and tagged to:
- shopping:latest
- shopping_admin:latest
- reddit:latest
- gitlab:latest
- openstreetmap-website-db:latest
- openstreetmap-website-web:latest

Ensure that the artifacts required for the Wikipedia environment `wikipedia_en_all_maxi_2022-05.zim`, and maps environment `openstreetmap-website.tar.gz` are present in [../../environment_setup/webarena/archive](../../environment_setup/webarena/archive)

Then, run
```sh
cd ~/amber
bash environment_setup/webarena/maps/setup.sh
bash environment_setup/webarena/wikipedia/setup.sh
```