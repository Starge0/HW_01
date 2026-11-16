# Image sources and licenses

The three photographs come from Wikimedia Commons, which mirrors them from Unsplash.
For each file, the Commons page states that the image was published on Unsplash **before
5 June 2017** and is released under the **Creative Commons CC0 1.0 Universal Public Domain
Dedication** (CC0 requires no attribution; it is given anyway).

| Repo file (`data/raw/`) | Title | Author | Date | Place | License | Source page |
|---|---|---|---|---|---|---|
| `foggy_countryside.jpg` | Foggy countryside (Unsplash) | Anton Repponen (repponen) | 5 June 2016 | Kula, United States | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Foggy_countryside_(Unsplash).jpg |
| `foggy_forest.jpg` | Foggy Green Forest (Unsplash) | Guy Bowden (guybowden) | 23 May 2016 | Morzine, France | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Foggy_Green_Forest_(Unsplash).jpg |
| `foggy_village.jpg` | Foggy mountain village (Unsplash) | Christian Bruno (ciciancam) | 18 May 2015 | Lizzola, Italy | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Foggy_mountain_village_(Unsplash).jpg |

Credits:
- Image by Anton Repponen, sourced from Wikimedia Commons, licensed under CC0 1.0.
- Image by Guy Bowden, sourced from Wikimedia Commons, licensed under CC0 1.0.
- Image by Christian Bruno, sourced from Wikimedia Commons, licensed under CC0 1.0.

The Unsplash page of each photo is linked in the "Source" row of its Commons page.

Processing: the files in `data/raw/` are the reduced-size versions served by Commons
(about 1920 px on the longest side; the village photo is 1932 x 2576 px). The pipeline
resizes them to 800 px on the longest side (`cv2.INTER_AREA`) into `data/original/`.
The images in `data/degraded/` are synthetic degradations of those originals.
