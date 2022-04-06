package com.simiacryptus.mindseye.art.util

trait ArtworkStyleGalleries {

  /**
   * Gallery class that contains a name and lists of low and high resolution images.
   *
   * @param name    Name of the gallery
   * @param lowRes  List of low resolution images
   * @param highRes List of high resolution images
   * @docgenVersion 9
   */
  protected sealed case class Gallery
  (
    name: String,
    lowRes: List[String],
    highRes: List[String]
  )

  final def Rembrandt = Gallery(
    name = "rembrandt_religious",
    lowRes = List(
      "https://uploads1.wikiart.org/images/rembrandt/the-stoning-of-st-stephen-1625.jpg",
      "https://uploads6.wikiart.org/images/rembrandt/balaam-s-ass-1626.jpg",
      "https://uploads6.wikiart.org/images/rembrandt/christ-driving-the-moneychangers-from-the-temple-1626.jpg",
      "https://uploads7.wikiart.org/images/rembrandt/the-rich-fool-1627.jpg",
      "https://uploads3.wikiart.org/images/rembrandt/samson-and-delilah-1628.jpg",
      "https://uploads4.wikiart.org/images/rembrandt/christ-on-the-cross-1631.jpg",
      "https://uploads7.wikiart.org/images/rembrandt/the-raising-of-lazarus-1630.jpg",
      "https://uploads8.wikiart.org/00142/images/rembrandt/christ-in-the-storm.jpg",
      "https://uploads6.wikiart.org/images/rembrandt/the-elevation-of-the-cross.jpg",
      "https://uploads5.wikiart.org/images/rembrandt/holy-family-1634.jpg",
      "https://uploads2.wikiart.org/images/rembrandt/susanna-and-the-elders.jpg",
      "https://uploads7.wikiart.org/images/rembrandt/the-blinding-of-samson-1636.jpg",
      "https://uploads1.wikiart.org/images/rembrandt/tobias-cured-with-his-son-1636.jpg",
      "https://uploads3.wikiart.org/images/rembrandt/the-archangel-raphael-taking-leave-of-the-tobit-family-1637.jpg",
      "https://uploads7.wikiart.org/00168/images/rembrandt/rembrandt-van-rijn-christ-and-st-mary-magdalen-at-the-tomb-google-art-project-1.jpg",
      "https://uploads6.wikiart.org/images/rembrandt/jacob-wrestling-with-the-angel-1659.jpg",
      "https://uploads7.wikiart.org/00199/images/rembrandt/woa-image-1-2.jpg"
    ),
    highRes = List(
      "https://uploads8.wikiart.org/00340/images/gerard-van-honthorst/adoration-of-the-shepherds.jpg",
      "https://uploads1.wikiart.org/00268/images/vincenzo-camuccini/vincenzo-camuccini-scipios-m-igung-2782-kunsthistorisches-museum.jpg"
    )
  )

  final def CubismPortraits = Gallery(
    name = "cubism_portraits",
    lowRes = List(
      "https://uploads3.wikiart.org/images/pablo-picasso/bust-of-young-woman-from-avignon-1907.jpg",
      "https://uploads7.wikiart.org/images/pablo-picasso/nude-bust-1907.jpg",
      "https://uploads5.wikiart.org/images/pablo-picasso/woman-with-a-fan-1907.jpg",
      "https://uploads8.wikiart.org/images/pablo-picasso/farm-woman-1908.jpg",
      "https://uploads3.wikiart.org/images/pablo-picasso/woman-with-yellow-shirt-1907.jpg",
      "https://uploads2.wikiart.org/images/pablo-picasso/head-of-a-man-1908.jpg",
      "https://uploads4.wikiart.org/images/pablo-picasso/head-of-a-man-1908-2.jpg",
      "https://uploads4.wikiart.org/images/pablo-picasso/head-of-a-man-1908-1.jpg",
      "https://uploads2.wikiart.org/images/pablo-picasso/head-of-woman-1908.jpg",
      "https://uploads5.wikiart.org/00323/images/pablo-picasso/ii1640-picasso.jpg",
      "https://uploads3.wikiart.org/images/pablo-picasso/queen-isabella-1908.jpg",
      "https://uploads5.wikiart.org/images/georges-braque/head-of-a-woman-1909.jpg",
      "https://uploads1.wikiart.org/images/pablo-picasso/bust-of-woman-with-flowers.jpg",
      "https://uploads0.wikiart.org/00234/images/pablo-picasso/pablo-picasso-femme-assise.jpg",
      "https://uploads7.wikiart.org/images/pablo-picasso/portrait-of-manuel-pallares-1909.jpg",
      "https://uploads7.wikiart.org/00323/images/pablo-picasso/head-of-a-woman-picasso-sculpture-pablo-picasso-s-women-wives-lovers-and-muses-of-head-of-a-woman.jpg",
      "https://uploads0.wikiart.org/00165/images/robert-falk/falk-portret-van-elizabeth-sergejevna-potehinoj-1910.jpg",
      "https://uploads0.wikiart.org/images/kazimir-malevich/the-reaper-on-red-1913.jpg",
      "https://uploads7.wikiart.org/images/kazimir-malevich/peasant-woman-with-buckets-and-a-child.jpg",
      "https://uploads8.wikiart.org/images/vladimir-tatlin/portrait-of-the-artist.jpg",
      "https://uploads5.wikiart.org/images/francis-picabia/young-girl.jpg",
      "https://uploads4.wikiart.org/images/pyotr-konchalovsky/portrait-of-the-artist-vladimir-rozhdestvensky-1912.jpg",
      "https://uploads5.wikiart.org/images/olga-rozanova/the-portrait-of-a-rozanov-1912.jpg",
      "https://uploads2.wikiart.org/images/man-ray/portrait-of-alfred-stieglitz-1913.jpg",
      "https://uploads5.wikiart.org/images/amadeo-de-souza-cardoso/head-1913-1.jpg",
      "https://uploads8.wikiart.org/00308/images/josef-capek/josef-apek-hlava-1913.jpg",
      "https://uploads4.wikiart.org/images/otto-dix/the-nun.jpg",
      "https://uploads3.wikiart.org/images/amadeo-de-souza-cardoso/sorrows-heads-1914.jpg",
      "https://uploads5.wikiart.org/images/alberto-magnelli/the-drunk-man-1914.jpg",
      "https://uploads3.wikiart.org/images/henri-matisse/head-white-and-pink.jpg",
      "https://uploads0.wikiart.org/images/diego-rivera/portrait-de-martin-luis-guzman-1915.jpg",
      "https://uploads4.wikiart.org/images/amadeo-de-souza-cardoso/green-eye-mask-1915.jpg",
      "https://uploads3.wikiart.org/images/amadeo-de-souza-cardoso/head-1915.jpg",
      "https://uploads5.wikiart.org/00308/images/josef-capek/josef-apek-hlava-v-napoleonsk-m-klobouku-1915.jpg",
      "https://uploads2.wikiart.org/00308/images/josef-capek/josef-apek-chlapec-v-m-kk-m-kllobouku-1915.jpg",
      "https://uploads3.wikiart.org/00308/images/josef-capek/josef-apek-hlava-asi-1915.jpg",
      "https://uploads2.wikiart.org/00308/images/josef-capek/josef-apek-mlad-ena-1915.jpg",
      "https://uploads4.wikiart.org/00308/images/josef-capek/josef-apek-podoba-z-biografu-1915-1.jpg",
      "https://uploads8.wikiart.org/00308/images/josef-capek/josef-apek-nev-stka-s-konvalinkami-1917.jpg",
      "https://uploads2.wikiart.org/00308/images/josef-capek/josef-apek-prodava-kohoutk-1917-1.jpg"
    ),
    highRes = List(
      "https://uploads1.wikiart.org/00198/images/pablo-picasso/old-guitarist-chicago.jpg",
      "https://uploads6.wikiart.org/00371/images/vincent-van-gogh/vincent-van-gogh-head-of-a-skeleton-with-a-burning-cigarette-google-art-project.jpg",
      "https://uploads6.wikiart.org/00334/images/maria-blanchard/femme-assise-seated-woman-c-1917-oil-on-canvas-1.jpg"
    )
  )

  final def RealismLandscapes = Gallery(
    name = "realism_landscapes",
    lowRes = List(
      "https://uploads6.wikiart.org/images/camille-corot/civita-castellana-1827.jpg",
      "https://uploads7.wikiart.org/images/camille-corot/civita-castellana-buildings-high-in-the-rocks-la-porta-san-salvatore.jpg",
      "https://uploads5.wikiart.org/images/camille-corot/mont-soracte.jpg",
      "https://uploads1.wikiart.org/images/camille-corot/olevano-the-town-and-the-rocks-1827.jpg",
      "https://uploads2.wikiart.org/images/camille-corot/aqueducts-in-the-roman-campagna-1828.jpg",
      "https://uploads3.wikiart.org/images/camille-corot/trees-and-rocks-at-la-serpentara-1827.jpg",
      "https://uploads6.wikiart.org/images/camille-corot/civita-castellana-and-mount-soracte-1826.jpg",
      "https://uploads4.wikiart.org/images/camille-corot/little-chaville-1825.jpg",
      "https://uploads5.wikiart.org/images/camille-corot/the-promenade-du-poussin-roman-campagna.jpg",
      "https://uploads7.wikiart.org/images/camille-corot/the-roman-campagna-with-the-claudian-aqueduct-1828.jpg",
      "https://uploads8.wikiart.org/images/camille-corot/the-tiber-near-rome-1828.jpg",
      "https://uploads8.wikiart.org/images/theodore-rousseau/mountain-landscape-with-fisherman-1830.jpg",
      "https://uploads7.wikiart.org/images/theodore-rousseau/not_detected_198929.jpg",
      "https://uploads7.wikiart.org/images/theodore-rousseau/valley-in-the-auvergne-mountains-1830.jpg",
      "https://uploads2.wikiart.org/images/theodore-rousseau/a-view-of-thiers-in-the-auvergne.jpg",
      "https://uploads1.wikiart.org/images/theodore-rousseau/water-mill-thiers-1830.jpg",
      "https://uploads2.wikiart.org/images/theodore-rousseau/landscape-at-vigerie-valley-santoire-auvergne-1830.jpg",
      "https://uploads6.wikiart.org/images/theodore-rousseau/mountain-stream-in-the-auvergne-1830.jpg",
      "https://uploads8.wikiart.org/images/theodore-rousseau/hilly-landscape-auvergne-1830.jpg",
      "https://uploads3.wikiart.org/images/camille-corot/fontainebleau-black-oaks-of-bas-breau.jpg",
      "https://uploads4.wikiart.org/images/camille-corot/in-the-woods-at-ville-d-avray.jpg",
      "https://uploads8.wikiart.org/images/mikhail-lebedev/ariccia-near-rome-1836.jpg",
      "https://uploads0.wikiart.org/images/mikhail-lebedev/an-alley-in-albano-1837.jpg",
      "https://uploads8.wikiart.org/images/camille-corot/lake-nemi-seen-through-trees-1843.jpg"
    ),
    highRes = List(
      "https://uploads7.wikiart.org/00364/images/carlo-ademollo/carlo-ademollo-travellers-in-the-mountains-1850.jpg",
      "https://uploads7.wikiart.org/00366/images/johan-christian-dahl/egmg1hmxsaadhx.jpg",
      "https://uploads4.wikiart.org/00367/images/johann-erdmann-hummel/schloss-wilhelmsh-he-mit-dem-habichtswald-1800.jpg",
      "https://uploads8.wikiart.org/00366/images/johan-christian-dahl/ex-wqckwqaarhjq.jpg"
    )
  )

  final def VanGogh = Gallery(
    name = "van_gough",
    lowRes = List(
      "https://uploads4.wikiart.org/images/vincent-van-gogh/windmils-at-dordrecht-1881.jpg",
      "https://uploads0.wikiart.org/images/vincent-van-gogh/edge-of-a-wood-1882(1).jpg",
      "https://uploads5.wikiart.org/images/vincent-van-gogh/iron-mill-in-the-hague-1882.jpg",
      "https://uploads5.wikiart.org/images/vincent-van-gogh/shores-of-scheveningen-1882.jpg",
      "https://uploads8.wikiart.org/images/vincent-van-gogh/farmhouses-among-trees-1883(1).jpg",
      "https://uploads3.wikiart.org/images/vincent-van-gogh/landscape-with-dunes-1883.jpg",
      "https://uploads6.wikiart.org/images/vincent-van-gogh/avenue-of-poplars-at-sunset-1884-1(1).jpg",
      "https://uploads0.wikiart.org/images/vincent-van-gogh/vase-with-honesty-1884.jpg",
      "https://uploads2.wikiart.org/images/vincent-van-gogh/backyards-of-old-houses-in-antwerp-in-the-snow-1885.jpg",
      "https://uploads2.wikiart.org/images/vincent-van-gogh/autumn-landscape-1885(1).jpg",
      "https://uploads4.wikiart.org/images/vincent-van-gogh/cottage-and-woman-with-goat-1885(1).jpg",
      "https://uploads3.wikiart.org/images/vincent-van-gogh/lane-with-poplars-near-nuenen-1885.jpg",
      "https://uploads0.wikiart.org/images/vincent-van-gogh/the-old-cemetery-tower-at-nuenen-in-the-snow-1885.jpg"
    ),
    highRes = List(
      "https://uploads8.wikiart.org/00175/images/vincent-van-gogh/starry-night-over-the-rhone.jpg",
      "https://uploads3.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg",
      "https://uploads8.wikiart.org/00213/images/vincent-van-gogh/antique-3840759.jpg",
      "https://uploads6.wikiart.org/00371/images/vincent-van-gogh/vincent-van-gogh-head-of-a-skeleton-with-a-burning-cigarette-google-art-project.jpg",
      "https://uploads4.wikiart.org/images/vincent-van-gogh/still-life-with-drawing-board-pipe-onions-and-sealing-wax-1889.jpg",
      "https://uploads6.wikiart.org/images/vincent-van-gogh/the-night-cafe-1888.jpg",
      "https://uploads8.wikiart.org/images/vincent-van-gogh/harvest-at-la-crau-with-montmajour-in-the-background-1888.jpg",
      "https://uploads8.wikiart.org/images/vincent-van-gogh/view-of-arles-with-irises-in-the-foreground-1888.jpg"
    )
  )

  private def galleries = List(
    Rembrandt,
    CubismPortraits,
    RealismLandscapes,
    VanGogh
  )

  /**
   * This function returns a sequence of strings that have been
   * filtered and mapped from a given list of strings.
   *
   * @docgenVersion 9
   */
  def styleGalleries_lowRes(list: Seq[String]): Seq[String] = list.flatMap(name => {
    galleries.find(_.name == name).map(_.lowRes).getOrElse(List(name))
  }).filter(_.nonEmpty)

  /**
   * Takes a list of gallery names and returns a list of high-resolution gallery names.
   *
   * @param list a sequence of gallery names
   * @return a sequence of high-resolution gallery names
   * @docgenVersion 9
   */
  def styleGalleries_highRes(list: Seq[String]): Seq[String] = list.flatMap(name => {
    galleries.find(_.name == name).map(_.highRes).getOrElse(List(name))
  }).filter(_.nonEmpty).filter(_.nonEmpty)

}
