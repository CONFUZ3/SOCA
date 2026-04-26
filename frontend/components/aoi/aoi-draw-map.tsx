"use client";

import "@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css";
import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import MapboxDraw from "@mapbox/mapbox-gl-draw";
import area from "@turf/area";

const BASEMAP =
  "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

const MIN_KM2 = 0.5;
const MAX_KM2 = 50_000;

// MapboxDraw default styles patched for MapLibre compatibility:
// array branches inside `case` expressions must be wrapped in ["literal", [...]]
const DRAW_STYLES: MapboxDraw.MapboxDrawOptions["styles"] = [
  {
    id: "gl-draw-polygon-fill",
    type: "fill",
    filter: ["all", ["==", "$type", "Polygon"]],
    paint: {
      "fill-color": ["case", ["==", ["get", "active"], "true"], "#fbb03b", "#3bb2d0"],
      "fill-opacity": 0.1,
    },
  },
  {
    id: "gl-draw-lines",
    type: "line",
    filter: ["any", ["==", "$type", "LineString"], ["==", "$type", "Polygon"]],
    layout: { "line-cap": "round", "line-join": "round" },
    paint: {
      "line-color": ["case", ["==", ["get", "active"], "true"], "#fbb03b", "#3bb2d0"],
      "line-dasharray": [
        "case",
        ["==", ["get", "active"], "true"], ["literal", [0.2, 2]],
        ["literal", [2, 0]],
      ],
      "line-width": 2,
    },
  },
  {
    id: "gl-draw-point-outer",
    type: "circle",
    filter: ["all", ["==", "$type", "Point"], ["==", "meta", "feature"]],
    paint: {
      "circle-radius": ["case", ["==", ["get", "active"], "true"], 7, 5],
      "circle-color": "#fff",
    },
  },
  {
    id: "gl-draw-point-inner",
    type: "circle",
    filter: ["all", ["==", "$type", "Point"], ["==", "meta", "feature"]],
    paint: {
      "circle-radius": ["case", ["==", ["get", "active"], "true"], 5, 3],
      "circle-color": ["case", ["==", ["get", "active"], "true"], "#fbb03b", "#3bb2d0"],
    },
  },
  {
    id: "gl-draw-vertex-outer",
    type: "circle",
    filter: ["all", ["==", "$type", "Point"], ["==", "meta", "vertex"], ["!=", "mode", "simple_select"]],
    paint: {
      "circle-radius": ["case", ["==", ["get", "active"], "true"], 7, 5],
      "circle-color": "#fff",
    },
  },
  {
    id: "gl-draw-vertex-inner",
    type: "circle",
    filter: ["all", ["==", "$type", "Point"], ["==", "meta", "vertex"], ["!=", "mode", "simple_select"]],
    paint: {
      "circle-radius": ["case", ["==", ["get", "active"], "true"], 5, 3],
      "circle-color": "#fbb03b",
    },
  },
  {
    id: "gl-draw-midpoint",
    type: "circle",
    filter: ["all", ["==", "meta", "midpoint"]],
    paint: { "circle-radius": 3, "circle-color": "#fbb03b" },
  },
];

interface Props {
  onDrawn: (geojson: GeoJSON.FeatureCollection, area_km2: number) => void;
  onAreaChange: (area_km2: number | null, error: string | null) => void;
  initialGeojson?: GeoJSON.FeatureCollection | null;
}

export function AoiDrawMap({ onDrawn, onAreaChange, initialGeojson }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const drawRef = useRef<MapboxDraw | null>(null);

  // Keep latest callbacks in refs so map event handlers always call the
  // current version without the effect needing to re-run.
  const onDrawnRef = useRef(onDrawn);
  const onAreaChangeRef = useRef(onAreaChange);
  useEffect(() => { onDrawnRef.current = onDrawn; }, [onDrawn]);
  useEffect(() => { onAreaChangeRef.current = onAreaChange; }, [onAreaChange]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP,
      center: [0, 20],
      zoom: 1.5,
      attributionControl: false,
      dragRotate: false,
      touchPitch: false,
    });

    map.addControl(
      new maplibregl.AttributionControl({ compact: true }),
      "bottom-right",
    );
    map.addControl(
      new maplibregl.NavigationControl({ showCompass: false }),
      "top-right",
    );

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: { polygon: true, trash: true },
      styles: DRAW_STYLES,
    });

    map.on("load", () => {
      map.addControl(draw as unknown as maplibregl.IControl);
      drawRef.current = draw;

      if (initialGeojson && initialGeojson.features.length > 0) {
        draw.add(initialGeojson);
        // Fit map to the loaded geometry
        const coords = (
          initialGeojson.features[0].geometry as GeoJSON.Polygon
        ).coordinates[0];
        if (coords.length >= 2) {
          const lngs = coords.map((c) => c[0]);
          const lats = coords.map((c) => c[1]);
          map.fitBounds(
            [
              [Math.min(...lngs), Math.min(...lats)],
              [Math.max(...lngs), Math.max(...lats)],
            ],
            { padding: 60, duration: 0 },
          );
        }
      } else {
        draw.changeMode("draw_polygon");
        map.getCanvas().style.cursor = "crosshair";
      }

      // MapboxDraw 1.5 / MapLibre GL 5 incompatibility: after a scroll-zoom
      // the draw_polygon mode silently stops registering clicks. Re-enter it
      // whenever a zoom or pan ends and no polygon has been placed yet.
      const reenterDrawIfNeeded = () => {
        if (draw.getAll().features.length === 0) {
          if (draw.getMode() !== "draw_polygon") {
            draw.changeMode("draw_polygon");
          }
          map.getCanvas().style.cursor = "crosshair";
        }
      };
      map.on("zoomend", reenterDrawIfNeeded);
      map.on("moveend", reenterDrawIfNeeded);

      // Once a polygon is committed, clear the crosshair cursor.
      map.on("draw.create", () => {
        map.getCanvas().style.cursor = "";
      });
      map.on("draw.delete", () => {
        draw.changeMode("draw_polygon");
        map.getCanvas().style.cursor = "crosshair";
      });
    });

    const handleChange = () => {
      const fc = draw.getAll();
      if (!fc.features.length) {
        onAreaChangeRef.current(null, null);
        return;
      }
      const m2 = area(fc as Parameters<typeof area>[0]);
      const km2 = m2 / 1e6;
      if (km2 < MIN_KM2) {
        onAreaChangeRef.current(
          km2,
          `Area too small — ${km2.toFixed(3)} km² (min ${MIN_KM2} km²)`,
        );
        return;
      }
      if (km2 > MAX_KM2) {
        onAreaChangeRef.current(
          km2,
          `Area too large — ${Math.round(km2).toLocaleString()} km² (max ${MAX_KM2.toLocaleString()} km²)`,
        );
        return;
      }
      onAreaChangeRef.current(km2, null);
      onDrawnRef.current(fc, km2);
    };

    const handleDelete = () => onAreaChangeRef.current(null, null);

    map.on("draw.create", handleChange);
    map.on("draw.update", handleChange);
    map.on("draw.delete", handleDelete);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      drawRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return <div ref={containerRef} className="h-full w-full" />;
}
