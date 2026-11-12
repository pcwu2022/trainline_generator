# Explicit water crossings

The fast generator supports configured crossing assets, priced with the legacy
optimizer's continuous-water growth formula. The Europe profile enables this mode
and supplies an approximate Channel crossing. Other profiles retain the previous
sea-blocking behavior by default.

```json
{
  "alignment": {
    "sea_crossings_enabled": true,
    "water_meur_km": 120,
    "water_growth_km": 25,
    "water_sample_km": 0.25,
    "water_crossings": [
      {"id": "channel", "geometry": [[1.12, 51.10], [1.80, 50.92]]}
    ],
    "max_capex_meur": 50000
  }
}
```

Geometry is a WGS84 polyline in longitude/latitude order. Portal cells must be on
land; move a portal inland if the coarse raster labels it water. Crossings outside
the routing extent are ignored. Exclusions, missing land terrain, and protected
classes remain forbidden. Known sea does not require valid land elevation data.
Country boundaries classify sea independently of the selected countries.

When enabled, ordinary routing stays on land and can traverse water only through
configured assets. This applies to inland water as well as sea: supply additional
assets for rivers or lakes that need crossings. Automatic coastal candidate
generation is not implemented. City-pair generation and local routing windows
still bound which connections can be discovered; an asset alone does not force a
city pair into the candidate graph. The geometry represents an abstract crossing,
not a detailed bridge or tunnel engineering design.

For each continuous water stretch, cost is its integrated baseline cost times
`exp(water_length_km / water_growth_km)`. The baseline is the larger of the track
base rate and `water_meur_km`, plus any urban surcharge. Land resets the exponent.
Land segments within the configured geometry use normal construction rates.
Geometry is sampled at no more than `water_sample_km`, splitting at raster cell
boundaries too. This does not restore coastlines or rivers absent from the source
grid. The exponent is capped at 100 for numerical stability, as in the old router.

Each asset becomes a virtual infrastructure tile. Its entire cost and its train
capacity are shared across corridors using that asset, using the existing track
layout rules. Its internal geometry is not charged again as ordinary raster
tiles; portal cells remain part of the connecting land corridors. Assets with
different IDs are treated as different physical infrastructure. Avoid overlapping
asset definitions that describe the same crossing.

The raster router is augmented with portal links and searches their full nonlinear
cost alongside land travel. Output geometry follows the priced crossing rather
than smoothing across water. `crossing_ids` and `water_km` are included in alignment
caches and corridor GeoJSON. Cache metadata records asset lengths and costs.
Cache version 8 invalidates old alignments; crossing settings and geometry are
included in the configuration fingerprint. Existing run snapshots are not
rewritten: start a fresh run to evaluate the changed candidate graph.

On the current European inputs, the example asset measures approximately 54.0 km,
including 46.3 km classified as water, and costs €35.5–35.7bn before city approaches.
This is a coarse-grid scenario cost, not a measured tunnel cost. The Europe
profile raises the total alignment cap from €25bn to €50bn so this scenario can
enter the candidate graph. London–Lille and London–Paris were both verified to
route through it. There is no Gotthard underground alignment model in this change.
