//! Generate SVG gallery images for the README.
//!
//! Run with: cargo run --example generate_gallery

use approxum::curves::{offset_cubic_to_polyline, CubicBezier2};
use approxum::polygon::{
    offset_polygon, polygon_difference, polygon_intersection, polygon_union,
    stroke_polyline_with_options, visibility_polygon_with_obstacles, JoinStyle, LineCap, LineJoin,
    Polygon, StrokeOptions,
};
use approxum::sampling::poisson_disk;
use approxum::triangulation::{delaunay_triangulation, voronoi_diagram, VoronoiEdge};
use approxum::Point2;
use std::f64::consts::PI;
use std::fs;

/// Simple SVG builder
struct Svg {
    content: String,
    view_box: Option<(f64, f64, f64, f64)>,
}

impl Svg {
    fn new() -> Self {
        Self {
            content: String::new(),
            view_box: None,
        }
    }

    fn with_viewbox(x: f64, y: f64, w: f64, h: f64) -> Self {
        let mut svg = Self::new();
        svg.view_box = Some((x, y, w, h));
        svg
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str) {
        self.content.push_str(&format!(
            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}"/>"#,
            x, y, w, h, fill
        ));
    }

    fn path(&mut self, d: &str, fill: &str, stroke: &str, stroke_width: f64) {
        self.content.push_str(&format!(
            r#"<path d="{}" fill="{}" stroke="{}" stroke-width="{}" fill-rule="nonzero"/>"#,
            d, fill, stroke, stroke_width
        ));
    }

    /// Path with outline drawn behind fill (for text that works on light and dark backgrounds)
    fn path_outlined(&mut self, d: &str, fill: &str, outline: &str, outline_width: f64) {
        self.content.push_str(&format!(
            r#"<path d="{}" fill="{}" stroke="{}" stroke-width="{}" stroke-linejoin="round" paint-order="stroke fill" fill-rule="nonzero"/>"#,
            d, fill, outline, outline_width
        ));
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str, stroke: &str, stroke_width: f64) {
        self.content.push_str(&format!(
            r#"<circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
            cx, cy, r, fill, stroke, stroke_width
        ));
    }

    fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, stroke_width: f64) {
        self.content.push_str(&format!(
            r#"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="{}" stroke-linecap="round"/>"#,
            x1, y1, x2, y2, stroke, stroke_width
        ));
    }

    fn polyline_path(&mut self, points: &[Point2<f64>], fill: &str, stroke: &str, stroke_width: f64) {
        if points.is_empty() {
            return;
        }
        let mut d = format!("M {:.2} {:.2}", points[0].x, points[0].y);
        for p in &points[1..] {
            d.push_str(&format!(" L {:.2} {:.2}", p.x, p.y));
        }
        self.path(&d, fill, stroke, stroke_width);
    }

    fn polygon(&mut self, points: &[Point2<f64>], fill: &str, stroke: &str, stroke_width: f64) {
        if points.is_empty() {
            return;
        }
        let mut d = format!("M {:.2} {:.2}", points[0].x, points[0].y);
        for p in &points[1..] {
            d.push_str(&format!(" L {:.2} {:.2}", p.x, p.y));
        }
        d.push_str(" Z");
        self.path(&d, fill, stroke, stroke_width);
    }

    /// Polygon with outline drawn behind fill (for shapes that work on any background)
    fn polygon_outlined(&mut self, points: &[Point2<f64>], fill: &str, outline: &str, outline_width: f64) {
        if points.is_empty() {
            return;
        }
        let mut d = format!("M {:.2} {:.2}", points[0].x, points[0].y);
        for p in &points[1..] {
            d.push_str(&format!(" L {:.2} {:.2}", p.x, p.y));
        }
        d.push_str(" Z");
        self.path_outlined(&d, fill, outline, outline_width);
    }

    fn text(&mut self, x: f64, y: f64, text: &str, size: f64, fill: &str) {
        self.content.push_str(&format!(
            r#"<text x="{:.2}" y="{:.2}" font-family="monospace" font-size="{:.1}" fill="{}">{}</text>"#,
            x, y, size, fill, text
        ));
    }

    fn to_string(&self, width: f64, height: f64) -> String {
        let vb = self
            .view_box
            .map(|(x, y, w, h)| format!("viewBox=\"{:.2} {:.2} {:.2} {:.2}\"", x, y, w, h))
            .unwrap_or_default();
        format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" {}>{}</svg>"#,
            width, height, vb, self.content
        )
    }
}

/// Generate the "approxum" logo using carefully designed Bézier curves
fn generate_logo() -> String {
    // Typography parameters
    let stroke_weight = 14.0;
    let x_height = 60.0;
    let descender = 24.0;
    let letter_gap = 8.0;

    let options = StrokeOptions {
        width: stroke_weight,
        line_cap: LineCap::Butt,
        line_join: LineJoin::Miter,
        miter_limit: 4.0,
        tolerance: 0.3,
    };

    // Collect all letter outlines
    let mut all_paths: Vec<Vec<Point2<f64>>> = Vec::new();
    let mut cursor_x = 0.0;
    let baseline = x_height + descender;

    // Helper function to render strokes with consistent CCW winding
    fn render_strokes(
        strokes: Vec<Vec<Point2<f64>>>,
        options: &StrokeOptions<f64>,
        all_paths: &mut Vec<Vec<Point2<f64>>>,
    ) {
        for stroke in strokes {
            if stroke.len() >= 2 {
                let outline = stroke_polyline_with_options(&stroke, options);
                if outline.vertices.len() >= 3 {
                    // Ensure CCW winding (positive area)
                    let mut verts = outline.vertices;
                    let area = polygon_area(&verts);
                    if area < 0.0 {
                        verts.reverse();
                    }
                    all_paths.push(verts);
                }
            }
        }
    }

    // Calculate signed polygon area
    fn polygon_area(verts: &[Point2<f64>]) -> f64 {
        let n = verts.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += verts[i].x * verts[j].y;
            area -= verts[j].x * verts[i].y;
        }
        area / 2.0
    }

    // ===== LETTER 'a' =====
    let x = cursor_x;
    let bowl_w = 42.0;
    // Bowl (full oval)
    let mut bowl: Vec<Point2<f64>> = Vec::new();
    let bowl_cx = x + bowl_w / 2.0;
    let bowl_cy = baseline - x_height / 2.0;
    let bowl_rx = bowl_w / 2.0 - stroke_weight / 4.0;
    let bowl_ry = x_height / 2.0 - stroke_weight / 4.0;
    for i in 0..=32 {
        let angle = 2.0 * PI * (i as f64) / 32.0;
        bowl.push(Point2::new(
            bowl_cx + bowl_rx * angle.cos(),
            bowl_cy + bowl_ry * angle.sin(),
        ));
    }
    // Stem on right
    let stem = vec![
        Point2::new(x + bowl_w, baseline - x_height),
        Point2::new(x + bowl_w, baseline),
    ];
    render_strokes(vec![bowl, stem], &options, &mut all_paths); cursor_x += bowl_w + stroke_weight / 2.0 + letter_gap;

    // ===== LETTER 'p' (first) =====
    let x = cursor_x;
    let p_bowl_w = 32.0;  // Width of bowl extending from stem
    let p_bowl_h = x_height * 0.85;
    // Stem with descender
    let stem = vec![
        Point2::new(x + stroke_weight / 2.0, baseline - x_height),
        Point2::new(x + stroke_weight / 2.0, baseline + descender),
    ];
    // Bowl (half ellipse on right, directly connected to stem)
    let bowl: Vec<Point2<f64>> = (0..=16)
        .map(|i| {
            let angle = -PI / 2.0 + PI * (i as f64) / 16.0;
            Point2::new(
                x + stroke_weight / 2.0 + p_bowl_w * angle.cos(),  // Starts at stem when cos=0
                baseline - x_height + p_bowl_h / 2.0 + (p_bowl_h / 2.0) * angle.sin(),
            )
        })
        .collect();
    render_strokes(vec![stem, bowl], &options, &mut all_paths); cursor_x += p_bowl_w + stroke_weight + letter_gap;

    // ===== LETTER 'p' (second) =====
    let x = cursor_x;
    let stem = vec![
        Point2::new(x + stroke_weight / 2.0, baseline - x_height),
        Point2::new(x + stroke_weight / 2.0, baseline + descender),
    ];
    let bowl: Vec<Point2<f64>> = (0..=16)
        .map(|i| {
            let angle = -PI / 2.0 + PI * (i as f64) / 16.0;
            Point2::new(
                x + stroke_weight / 2.0 + p_bowl_w * angle.cos(),  // Starts at stem when cos=0
                baseline - x_height + p_bowl_h / 2.0 + (p_bowl_h / 2.0) * angle.sin(),
            )
        })
        .collect();
    render_strokes(vec![stem, bowl], &options, &mut all_paths); cursor_x += p_bowl_w + stroke_weight + letter_gap;

    // ===== LETTER 'r' =====
    let x = cursor_x;
    let r_width = 28.0;
    // Stem
    let stem = vec![
        Point2::new(x + stroke_weight / 2.0, baseline - x_height),
        Point2::new(x + stroke_weight / 2.0, baseline),
    ];
    // Shoulder (partial arc going right and up)
    let shoulder: Vec<Point2<f64>> = (0..=8)
        .map(|i| {
            let angle = PI / 2.0 - (PI / 2.0) * (i as f64) / 8.0;
            Point2::new(
                x + stroke_weight / 2.0 + r_width * (1.0 - angle.cos()),
                baseline - x_height + r_width * (1.0 - angle.sin()),
            )
        })
        .collect();
    render_strokes(vec![stem, shoulder], &options, &mut all_paths); cursor_x += r_width + stroke_weight / 2.0 + letter_gap;

    // ===== LETTER 'o' =====
    let x = cursor_x;
    let o_w = 44.0;
    let o_cx = x + o_w / 2.0;
    let o_cy = baseline - x_height / 2.0;
    let o_rx = o_w / 2.0 - stroke_weight / 4.0;
    let o_ry = x_height / 2.0 - stroke_weight / 4.0;
    let oval: Vec<Point2<f64>> = (0..=32)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / 32.0;
            Point2::new(o_cx + o_rx * angle.cos(), o_cy + o_ry * angle.sin())
        })
        .collect();
    render_strokes(vec![oval], &options, &mut all_paths); cursor_x += o_w + letter_gap;

    // ===== LETTER 'x' =====
    let x = cursor_x;
    let x_w = 38.0;
    // Two diagonal strokes
    let diag1 = vec![
        Point2::new(x, baseline - x_height),
        Point2::new(x + x_w, baseline),
    ];
    let diag2 = vec![
        Point2::new(x + x_w, baseline - x_height),
        Point2::new(x, baseline),
    ];
    render_strokes(vec![diag1, diag2], &options, &mut all_paths); cursor_x += x_w + letter_gap;

    // ===== LETTER 'u' =====
    let x = cursor_x;
    let u_w = 42.0;
    // U-shape: left stem down, curve at bottom, right stem up and down
    let u_curve: Vec<Point2<f64>> = (0..=16)
        .map(|i| {
            let angle = PI + PI * (i as f64) / 16.0;
            Point2::new(
                x + u_w / 2.0 + (u_w / 2.0 - stroke_weight / 3.0) * angle.cos(),
                baseline - (u_w / 2.0) + (u_w / 2.0 - stroke_weight / 3.0) * angle.sin().abs(),
            )
        })
        .collect();
    // Left vertical down to curve
    let left_stem = vec![
        Point2::new(x + stroke_weight / 3.0, baseline - x_height),
        Point2::new(x + stroke_weight / 3.0, baseline - u_w / 2.0),
    ];
    // Right vertical from curve up, then small tail down
    let right_stem = vec![
        Point2::new(x + u_w - stroke_weight / 3.0, baseline - u_w / 2.0),
        Point2::new(x + u_w - stroke_weight / 3.0, baseline - x_height),
    ];
    let right_tail = vec![
        Point2::new(x + u_w - stroke_weight / 3.0, baseline - x_height),
        Point2::new(x + u_w - stroke_weight / 3.0, baseline),
    ];
    render_strokes(vec![left_stem, u_curve, right_stem, right_tail], &options, &mut all_paths); cursor_x += u_w + letter_gap;

    // ===== LETTER 'm' =====
    let x = cursor_x;
    let m_w = 62.0;
    let hump_w = (m_w - stroke_weight) / 2.0;
    // Left stem (full height)
    let left_stem = vec![
        Point2::new(x + stroke_weight / 2.0, baseline - x_height),
        Point2::new(x + stroke_weight / 2.0, baseline),
    ];
    // First arch (goes up from stem, arcs over, comes down)
    let arch1: Vec<Point2<f64>> = (0..=12)
        .map(|i| {
            let angle = PI * (i as f64) / 12.0;  // 0 to PI
            Point2::new(
                x + stroke_weight / 2.0 + (hump_w / 2.0) * (1.0 - angle.cos()),
                baseline - x_height * (angle.sin()),  // goes up then down
            )
        })
        .collect();
    // Middle stem (from bottom of arch to baseline)
    let mid_stem = vec![
        Point2::new(x + stroke_weight / 2.0 + hump_w, baseline - x_height),
        Point2::new(x + stroke_weight / 2.0 + hump_w, baseline),
    ];
    // Second arch
    let arch2: Vec<Point2<f64>> = (0..=12)
        .map(|i| {
            let angle = PI * (i as f64) / 12.0;
            Point2::new(
                x + stroke_weight / 2.0 + hump_w + (hump_w / 2.0) * (1.0 - angle.cos()),
                baseline - x_height * (angle.sin()),
            )
        })
        .collect();
    // Right stem
    let right_stem = vec![
        Point2::new(x + stroke_weight / 2.0 + hump_w * 2.0, baseline - x_height),
        Point2::new(x + stroke_weight / 2.0 + hump_w * 2.0, baseline),
    ];
    render_strokes(
        vec![left_stem, arch1, mid_stem, arch2, right_stem],
        &options,
        &mut all_paths,
    );

    // Calculate bounding box
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for path in &all_paths {
        for p in path {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
    }

    // Add padding
    let padding = 12.0;
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;

    let width = max_x - min_x;
    let height = max_y - min_y;

    // Create SVG with tight viewBox (transparent background for light/dark theme support)
    let mut svg = Svg::with_viewbox(min_x, min_y, width, height);
    // No background rect - transparent

    // Render all paths with white outline for dark theme visibility
    // paint-order="stroke fill" draws outline behind the fill
    for path in &all_paths {
        svg.polygon_outlined(path, "#1a1a2e", "#ffffff", 3.0);
    }

    svg.to_string(width * 2.0, height * 2.0)
}

/// Visibility polygon example
fn generate_visibility() -> String {
    let mut svg = Svg::with_viewbox(0.0, 0.0, 400.0, 300.0);
    svg.rect(0.0, 0.0, 400.0, 300.0, "#0f0f23");

    // Room with obstacles
    let room = vec![
        Point2::new(20.0, 20.0),
        Point2::new(380.0, 20.0),
        Point2::new(380.0, 280.0),
        Point2::new(20.0, 280.0),
    ];

    // Obstacles
    let obstacles = vec![
        // L-shaped obstacle
        vec![
            Point2::new(80.0, 60.0),
            Point2::new(150.0, 60.0),
            Point2::new(150.0, 100.0),
            Point2::new(110.0, 100.0),
            Point2::new(110.0, 160.0),
            Point2::new(80.0, 160.0),
        ],
        // Triangle
        vec![
            Point2::new(200.0, 180.0),
            Point2::new(280.0, 220.0),
            Point2::new(200.0, 260.0),
        ],
        // Rectangle
        vec![
            Point2::new(300.0, 80.0),
            Point2::new(350.0, 80.0),
            Point2::new(350.0, 150.0),
            Point2::new(300.0, 150.0),
        ],
        // Small square
        vec![
            Point2::new(160.0, 200.0),
            Point2::new(190.0, 200.0),
            Point2::new(190.0, 230.0),
            Point2::new(160.0, 230.0),
        ],
    ];

    // Viewpoint
    let viewpoint = Point2::new(200.0, 120.0);

    // Calculate visibility polygon
    let room_poly = Polygon::new(room.clone());
    let obstacle_polys: Vec<Polygon<f64>> =
        obstacles.iter().map(|o| Polygon::new(o.clone())).collect();
    let vis = visibility_polygon_with_obstacles(&room_poly, &obstacle_polys, viewpoint);

    // Draw visibility area (lit region)
    svg.polygon(&vis.vertices, "rgba(255,215,0,0.3)", "#ffd700", 1.0);

    // Draw room boundary
    svg.polygon(&room, "none", "#4a4a6a", 2.0);

    // Draw obstacles
    for obs in &obstacles {
        svg.polygon(obs, "#2a2a4a", "#6a6a8a", 1.5);
    }

    // Draw light rays from viewpoint to visibility polygon vertices
    for v in vis.vertices.iter().take(20) {
        svg.line(viewpoint.x, viewpoint.y, v.x, v.y, "rgba(255,215,0,0.15)", 0.5);
    }

    // Draw viewpoint (light source)
    svg.circle(viewpoint.x, viewpoint.y, 8.0, "#ffd700", "#fff", 2.0);
    svg.circle(viewpoint.x, viewpoint.y, 4.0, "#fff", "none", 0.0);

    svg.to_string(400.0, 300.0)
}

/// Boolean operations example
fn generate_boolean_ops() -> String {
    let mut svg = Svg::with_viewbox(0.0, 0.0, 600.0, 250.0);
    svg.rect(0.0, 0.0, 600.0, 250.0, "white");

    // Create two overlapping shapes
    let circle1: Vec<Point2<f64>> = (0..32)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / 32.0;
            Point2::new(70.0 + 50.0 * angle.cos(), 125.0 + 50.0 * angle.sin())
        })
        .collect();

    let circle2: Vec<Point2<f64>> = (0..32)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / 32.0;
            Point2::new(110.0 + 50.0 * angle.cos(), 125.0 + 50.0 * angle.sin())
        })
        .collect();

    let poly1 = Polygon::new(circle1.clone());
    let poly2 = Polygon::new(circle2.clone());

    // Union
    let union_result = polygon_union(&poly1, &poly2);
    let offset_x = 0.0;
    for poly in &union_result {
        let shifted: Vec<_> = poly
            .vertices
            .iter()
            .map(|p| Point2::new(p.x + offset_x, p.y))
            .collect();
        svg.polygon(&shifted, "#3498db", "#2980b9", 2.0);
    }
    svg.text(60.0, 230.0, "Union", 14.0, "#333");

    // Intersection
    let inter_result = polygon_intersection(&poly1, &poly2);
    let offset_x = 170.0;
    // Draw original shapes faded
    let c1: Vec<_> = circle1
        .iter()
        .map(|p| Point2::new(p.x + offset_x, p.y))
        .collect();
    let c2: Vec<_> = circle2
        .iter()
        .map(|p| Point2::new(p.x + offset_x, p.y))
        .collect();
    svg.polygon(&c1, "rgba(52,152,219,0.2)", "#bdc3c7", 1.0);
    svg.polygon(&c2, "rgba(52,152,219,0.2)", "#bdc3c7", 1.0);
    for poly in &inter_result {
        let shifted: Vec<_> = poly
            .vertices
            .iter()
            .map(|p| Point2::new(p.x + offset_x, p.y))
            .collect();
        svg.polygon(&shifted, "#e74c3c", "#c0392b", 2.0);
    }
    svg.text(210.0, 230.0, "Intersection", 14.0, "#333");

    // Difference
    let diff_result = polygon_difference(&poly1, &poly2);
    let offset_x = 370.0;
    let c2: Vec<_> = circle2
        .iter()
        .map(|p| Point2::new(p.x + offset_x, p.y))
        .collect();
    svg.polygon(&c2, "rgba(52,152,219,0.2)", "#bdc3c7", 1.0);
    for poly in &diff_result {
        let shifted: Vec<_> = poly
            .vertices
            .iter()
            .map(|p| Point2::new(p.x + offset_x, p.y))
            .collect();
        svg.polygon(&shifted, "#2ecc71", "#27ae60", 2.0);
    }
    svg.text(410.0, 230.0, "Difference", 14.0, "#333");

    svg.to_string(600.0, 250.0)
}

/// Voronoi and Delaunay example
fn generate_voronoi() -> String {
    let mut svg = Svg::with_viewbox(0.0, 0.0, 400.0, 300.0);
    svg.rect(0.0, 0.0, 400.0, 300.0, "#1a1a2e");

    // Generate points using Poisson disk sampling
    let points = poisson_disk(360.0, 260.0, 35.0, 30);
    let points: Vec<_> = points
        .into_iter()
        .map(|p| Point2::new(p.x + 20.0, p.y + 20.0))
        .collect();

    // Delaunay triangulation
    let triangles = delaunay_triangulation(&points);

    // Draw Delaunay edges
    for tri in &triangles {
        let p0 = points[tri.a];
        let p1 = points[tri.b];
        let p2 = points[tri.c];
        svg.line(p0.x, p0.y, p1.x, p1.y, "rgba(52,152,219,0.4)", 1.0);
        svg.line(p1.x, p1.y, p2.x, p2.y, "rgba(52,152,219,0.4)", 1.0);
        svg.line(p2.x, p2.y, p0.x, p0.y, "rgba(52,152,219,0.4)", 1.0);
    }

    // Voronoi diagram
    let voronoi = voronoi_diagram(&points);

    // Draw Voronoi edges
    for edge in &voronoi.edges {
        match edge {
            VoronoiEdge::Finite(a, b) => {
                let pa = voronoi.vertices[*a];
                let pb = voronoi.vertices[*b];
                svg.line(pa.x, pa.y, pb.x, pb.y, "#e74c3c", 1.5);
            }
            VoronoiEdge::Ray(a, dir) => {
                let pa = voronoi.vertices[*a];
                // Extend ray to edge of viewport
                let pb = Point2::new(pa.x + dir.x * 500.0, pa.y + dir.y * 500.0);
                svg.line(pa.x, pa.y, pb.x, pb.y, "#e74c3c", 1.5);
            }
        }
    }

    // Draw points
    for p in &points {
        svg.circle(p.x, p.y, 3.0, "#ecf0f1", "none", 0.0);
    }

    svg.to_string(400.0, 300.0)
}

/// Polygon offset example with L-shape
fn generate_skeleton() -> String {
    let mut svg = Svg::with_viewbox(0.0, 0.0, 400.0, 300.0);
    svg.rect(0.0, 0.0, 400.0, 300.0, "white");

    // Create a convex hexagon centered in the viewport
    // (Convex shapes work correctly with the offset algorithm)
    let cx = 200.0;
    let cy = 150.0;
    let radius = 100.0;
    let shape: Vec<Point2<f64>> = (0..6)
        .map(|i| {
            let angle = -PI / 2.0 + (i as f64) * PI / 3.0; // Start from top
            Point2::new(cx + radius * angle.cos(), cy + radius * angle.sin())
        })
        .collect();

    let poly = Polygon::new(shape.clone());

    // Draw offset contours at various distances (inward)
    // Keep offsets small enough to avoid collapse/inversion at center
    let colors = ["#3498db", "#9b59b6", "#e74c3c", "#f39c12"];
    for (i, &dist) in [-15.0, -30.0, -40.0, -48.0].iter().enumerate() {
        let offset = offset_polygon(&poly, dist, JoinStyle::Miter, 4.0);
        if offset.vertices.len() >= 3 {
            svg.polygon(&offset.vertices, "none", colors[i], 2.0);
        }
    }

    // Draw original polygon
    svg.polygon(&shape, "none", "#2c3e50", 3.0);

    // Draw vertices
    for p in &shape {
        svg.circle(p.x, p.y, 4.0, "#2c3e50", "#fff", 1.5);
    }

    svg.to_string(400.0, 300.0)
}

/// Curve offset example
fn generate_curve_offset() -> String {
    let mut svg = Svg::with_viewbox(0.0, 0.0, 400.0, 250.0);
    svg.rect(0.0, 0.0, 400.0, 250.0, "#f8f9fa");

    // Create an S-curve
    let curve = CubicBezier2::new(
        Point2::new(50.0, 200.0),
        Point2::new(100.0, 50.0),
        Point2::new(300.0, 200.0),
        Point2::new(350.0, 50.0),
    );

    // Draw offset curves at various distances
    let offsets = [-40.0, -25.0, -12.0, 12.0, 25.0, 40.0];
    let colors = [
        "#9b59b6", "#3498db", "#1abc9c", "#1abc9c", "#3498db", "#9b59b6",
    ];

    for (&offset, &color) in offsets.iter().zip(colors.iter()) {
        let offset_curve = offset_cubic_to_polyline(&curve, offset, 0.5);
        svg.polyline_path(&offset_curve, "none", color, 2.0);
    }

    // Draw original curve (thicker)
    let main_curve = curve.to_polyline(0.5);
    svg.polyline_path(&main_curve, "none", "#2c3e50", 3.0);

    // Draw control points and handles
    svg.line(curve.p0.x, curve.p0.y, curve.p1.x, curve.p1.y, "#e74c3c", 1.0);
    svg.line(curve.p2.x, curve.p2.y, curve.p3.x, curve.p3.y, "#e74c3c", 1.0);
    svg.circle(curve.p0.x, curve.p0.y, 5.0, "#2c3e50", "#fff", 1.5);
    svg.circle(curve.p1.x, curve.p1.y, 4.0, "#e74c3c", "#fff", 1.5);
    svg.circle(curve.p2.x, curve.p2.y, 4.0, "#e74c3c", "#fff", 1.5);
    svg.circle(curve.p3.x, curve.p3.y, 5.0, "#2c3e50", "#fff", 1.5);

    svg.to_string(400.0, 250.0)
}

fn main() {
    // Create screenshots directory if it doesn't exist
    fs::create_dir_all("screenshots").expect("Failed to create screenshots directory");

    // Generate and save all images
    let images = [
        ("logo", generate_logo()),
        ("visibility", generate_visibility()),
        ("boolean", generate_boolean_ops()),
        ("voronoi", generate_voronoi()),
        ("skeleton", generate_skeleton()),
        ("curve_offset", generate_curve_offset()),
    ];

    for (name, svg) in images {
        let path = format!("screenshots/{}.svg", name);
        fs::write(&path, svg).expect(&format!("Failed to write {}", path));
        println!("Generated {}", path);
    }

    println!("\nAll gallery images generated successfully!");
}
