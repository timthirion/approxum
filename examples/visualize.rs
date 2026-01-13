//! Generates SVG visualizations for the README gallery.
//!
//! Run with: cargo run --example visualize

use approxum::bounds::Aabb2;
use approxum::curves::{Arc2, CubicBezier2};
use approxum::distance::{sdf_polygon, Circle, Sdf2};
use approxum::sampling::poisson_disk;
use approxum::simplify::{rdp, visvalingam};
use approxum::spatial::Bvh;
use approxum::Point2;

#[allow(unused_variables)]
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

const WIDTH: f64 = 800.0;
const HEIGHT: f64 = 400.0;

fn main() {
    generate_simplification();
    generate_curves();
    generate_poisson();
    generate_bvh();
    generate_sdf();
    println!("Generated all screenshots in screenshots/");
}

/// SVG helper to create an SVG document
struct Svg {
    content: String,
    width: f64,
    height: f64,
}

impl Svg {
    fn new(width: f64, height: f64) -> Self {
        Self {
            content: String::new(),
            width,
            height,
        }
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, stroke: &str, stroke_width: f64) {
        self.content.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
            x, y, w, h, fill, stroke, stroke_width
        ));
        self.content.push('\n');
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str, stroke: &str, stroke_width: f64) {
        self.content.push_str(&format!(
            r#"<circle cx="{}" cy="{}" r="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
            cx, cy, r, fill, stroke, stroke_width
        ));
        self.content.push('\n');
    }

    fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, stroke_width: f64) {
        self.content.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>"#,
            x1, y1, x2, y2, stroke, stroke_width
        ));
        self.content.push('\n');
    }

    fn polyline(&mut self, points: &[Point2<f64>], stroke: &str, stroke_width: f64, fill: &str) {
        let pts: String = points
            .iter()
            .map(|p| format!("{:.2},{:.2}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" ");
        self.content.push_str(&format!(
            r#"<polyline points="{}" fill="{}" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>"#,
            pts, fill, stroke, stroke_width
        ));
        self.content.push('\n');
    }

    fn polygon(&mut self, points: &[Point2<f64>], fill: &str, stroke: &str, stroke_width: f64) {
        let pts: String = points
            .iter()
            .map(|p| format!("{:.2},{:.2}", p.x, p.y))
            .collect::<Vec<_>>()
            .join(" ");
        self.content.push_str(&format!(
            r#"<polygon points="{}" fill="{}" stroke="{}" stroke-width="{}"/>"#,
            pts, fill, stroke, stroke_width
        ));
        self.content.push('\n');
    }

    fn text(&mut self, x: f64, y: f64, text: &str, font_size: f64, fill: &str) {
        self.content.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="system-ui, sans-serif" font-size="{}" fill="{}">{}</text>"#,
            x, y, font_size, fill, text
        ));
        self.content.push('\n');
    }

    fn group_start(&mut self, transform: &str) {
        self.content
            .push_str(&format!(r#"<g transform="{}">"#, transform));
        self.content.push('\n');
    }

    fn group_end(&mut self) {
        self.content.push_str("</g>\n");
    }

    fn save(&self, path: &str) {
        let svg = format!(
            r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">
<rect width="100%" height="100%" fill="#1a1a2e"/>
{}
</svg>"##,
            self.width, self.height, self.width, self.height, self.content
        );
        let mut file = File::create(path).unwrap();
        file.write_all(svg.as_bytes()).unwrap();
    }
}

/// Generate a noisy sine wave for simplification demos
fn generate_noisy_curve(num_points: usize, noise_scale: f64, seed: u64) -> Vec<Point2<f64>> {
    let mut state = seed;
    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1) as f64;
            let x = t * 350.0 + 25.0;
            // Center vertically with larger amplitude
            let base_y = 200.0 + (t * 4.0 * PI).sin() * 120.0;

            // Deterministic noise
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = ((state as f64 / u64::MAX as f64) - 0.5) * noise_scale;

            Point2::new(x, base_y + noise)
        })
        .collect()
}

fn generate_simplification() {
    let mut svg = Svg::new(WIDTH, HEIGHT);

    // Generate original noisy curve
    let original = generate_noisy_curve(200, 30.0, 12345);

    // Left panel: RDP
    svg.text(100.0, 30.0, "Ramer-Douglas-Peucker", 16.0, "#e0e0e0");

    // Original (faded)
    svg.polyline(&original, "#4a4a6a", 1.0, "none");

    // Simplified
    let rdp_result = rdp(&original, 8.0);
    svg.polyline(&rdp_result, "#00d4ff", 2.0, "none");

    // Points
    for p in &rdp_result {
        svg.circle(p.x, p.y, 3.0, "#00d4ff", "none", 0.0);
    }

    svg.text(
        25.0,
        370.0,
        &format!("{} → {} points", original.len(), rdp_result.len()),
        12.0,
        "#808080",
    );

    // Right panel: Visvalingam (draw directly at offset position)
    svg.text(500.0, 30.0, "Visvalingam-Whyatt", 16.0, "#e0e0e0");

    // Original (faded) - shift right by 400
    let original_shifted: Vec<_> = original.iter().map(|p| Point2::new(p.x + 400.0, p.y)).collect();
    svg.polyline(&original_shifted, "#4a4a6a", 1.0, "none");

    // Simplified
    let vis_result = visvalingam(&original, 200.0);
    let vis_shifted: Vec<_> = vis_result.iter().map(|p| Point2::new(p.x + 400.0, p.y)).collect();
    svg.polyline(&vis_shifted, "#ff6b6b", 2.0, "none");

    // Points
    for p in &vis_shifted {
        svg.circle(p.x, p.y, 3.0, "#ff6b6b", "none", 0.0);
    }

    svg.text(
        425.0,
        370.0,
        &format!("{} → {} points", original.len(), vis_result.len()),
        12.0,
        "#808080",
    );

    // Legend at bottom
    svg.text(250.0, 390.0, "Original curve (gray) vs Simplified (colored)", 12.0, "#606060");

    svg.save("screenshots/simplification.svg");
    println!("Generated simplification.svg");
}

fn generate_curves() {
    let mut svg = Svg::new(WIDTH, HEIGHT);

    // Left: Cubic Bézier
    svg.text(120.0, 30.0, "Cubic Bézier Curve", 16.0, "#e0e0e0");

    let p0 = Point2::new(50.0, 300.0);
    let p1 = Point2::new(100.0, 50.0);
    let p2 = Point2::new(300.0, 50.0);
    let p3 = Point2::new(350.0, 300.0);

    let bezier = CubicBezier2::new(p0, p1, p2, p3);
    let curve_points = bezier.to_polyline(1.0);

    // Control polygon (dashed effect via opacity)
    svg.line(p0.x, p0.y, p1.x, p1.y, "#4a4a6a", 1.5);
    svg.line(p1.x, p1.y, p2.x, p2.y, "#4a4a6a", 1.5);
    svg.line(p2.x, p2.y, p3.x, p3.y, "#4a4a6a", 1.5);

    // Curve
    svg.polyline(&curve_points, "#00d4ff", 3.0, "none");

    // Control points
    svg.circle(p0.x, p0.y, 6.0, "#ff6b6b", "#ffffff", 2.0);
    svg.circle(p1.x, p1.y, 6.0, "#ffd93d", "#ffffff", 2.0);
    svg.circle(p2.x, p2.y, 6.0, "#ffd93d", "#ffffff", 2.0);
    svg.circle(p3.x, p3.y, 6.0, "#ff6b6b", "#ffffff", 2.0);

    // Labels
    svg.text(p0.x - 15.0, p0.y + 20.0, "P₀", 12.0, "#808080");
    svg.text(p1.x - 15.0, p1.y - 10.0, "P₁", 12.0, "#808080");
    svg.text(p2.x + 5.0, p2.y - 10.0, "P₂", 12.0, "#808080");
    svg.text(p3.x + 5.0, p3.y + 20.0, "P₃", 12.0, "#808080");

    // Right: Arc
    svg.group_start("translate(400, 0)");
    svg.text(120.0, 30.0, "Circular Arc", 16.0, "#e0e0e0");

    let arc_start = Point2::new(50.0, 250.0);
    let arc_mid = Point2::new(200.0, 80.0);
    let arc_end = Point2::new(350.0, 250.0);

    if let Some(arc) = Arc2::from_three_points(arc_start, arc_mid, arc_end) {
        let arc_points = arc.to_polyline(1.0);
        svg.polyline(&arc_points, "#6bcb77", 3.0, "none");

        // Center and radius
        svg.circle(arc.center.x, arc.center.y, 4.0, "#ffd93d", "none", 0.0);
        svg.line(
            arc.center.x,
            arc.center.y,
            arc_start.x,
            arc_start.y,
            "#4a4a6a",
            1.0,
        );

        svg.text(arc.center.x + 10.0, arc.center.y, "center", 10.0, "#808080");
    }

    // Three defining points
    svg.circle(arc_start.x, arc_start.y, 5.0, "#ff6b6b", "#ffffff", 2.0);
    svg.circle(arc_mid.x, arc_mid.y, 5.0, "#ff6b6b", "#ffffff", 2.0);
    svg.circle(arc_end.x, arc_end.y, 5.0, "#ff6b6b", "#ffffff", 2.0);

    svg.group_end();

    svg.text(280.0, 380.0, "Control points (yellow) • Endpoints (red)", 12.0, "#606060");

    svg.save("screenshots/curves.svg");
    println!("Generated curves.svg");
}

fn generate_poisson() {
    let mut svg = Svg::new(WIDTH, HEIGHT);

    svg.text(320.0, 30.0, "Poisson Disk Sampling", 16.0, "#e0e0e0");

    // Generate points with different densities
    let scale = 3.5;

    // Left side: dense
    let points_dense: Vec<Point2<f64>> = poisson_disk(100.0, 100.0, 5.0, 30);
    for p in &points_dense {
        let x = p.x * scale + 30.0;
        let y = p.y * scale + 40.0;
        svg.circle(x, y, 3.0, "#00d4ff", "none", 0.0);
    }

    // Right side: sparse
    let points_sparse: Vec<Point2<f64>> = poisson_disk(100.0, 100.0, 10.0, 30);
    for p in &points_sparse {
        let x = p.x * scale + 430.0;
        let y = p.y * scale + 40.0;
        svg.circle(x, y, 4.0, "#6bcb77", "none", 0.0);
    }

    // Dividing line
    svg.line(400.0, 40.0, 400.0, 390.0, "#3a3a5a", 1.0);

    // Labels
    svg.text(120.0, 380.0, &format!("min_distance = 5 ({} points)", points_dense.len()), 12.0, "#808080");
    svg.text(520.0, 380.0, &format!("min_distance = 10 ({} points)", points_sparse.len()), 12.0, "#808080");

    svg.save("screenshots/poisson.svg");
    println!("Generated poisson.svg");
}

fn generate_bvh() {
    let mut svg = Svg::new(WIDTH, HEIGHT);

    svg.text(300.0, 30.0, "Bounding Volume Hierarchy", 16.0, "#e0e0e0");

    // Generate some random AABBs in clusters to show hierarchy better
    let mut aabbs = Vec::new();
    let mut state = 54321u64;

    // Cluster 1 (top-left)
    for _ in 0..8 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f64 / u64::MAX as f64) * 150.0 + 50.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f64 / u64::MAX as f64) * 120.0 + 50.0;

        let w = 25.0;
        let h = 25.0;
        aabbs.push(Aabb2::new(Point2::new(x, y), Point2::new(x + w, y + h)));
    }

    // Cluster 2 (top-right)
    for _ in 0..8 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f64 / u64::MAX as f64) * 150.0 + 450.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f64 / u64::MAX as f64) * 120.0 + 50.0;

        let w = 25.0;
        let h = 25.0;
        aabbs.push(Aabb2::new(Point2::new(x, y), Point2::new(x + w, y + h)));
    }

    // Cluster 3 (bottom-left)
    for _ in 0..7 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f64 / u64::MAX as f64) * 150.0 + 50.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f64 / u64::MAX as f64) * 120.0 + 200.0;

        let w = 25.0;
        let h = 25.0;
        aabbs.push(Aabb2::new(Point2::new(x, y), Point2::new(x + w, y + h)));
    }

    // Cluster 4 (bottom-right)
    for _ in 0..7 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as f64 / u64::MAX as f64) * 150.0 + 450.0;

        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let y = (state as f64 / u64::MAX as f64) * 120.0 + 200.0;

        let w = 25.0;
        let h = 25.0;
        aabbs.push(Aabb2::new(Point2::new(x, y), Point2::new(x + w, y + h)));
    }

    // Build BVH (we won't visualize internal structure, just show conceptual hierarchy)
    let _bvh = Bvh::build(&aabbs, 2);

    // Draw conceptual hierarchy bounds (manually computed to show idea)
    // Root level (encompasses all)
    svg.rect(40.0, 40.0, 720.0, 310.0, "#ff6b6b20", "#ff6b6b", 2.0);

    // Level 1: Left and right halves
    svg.rect(45.0, 45.0, 260.0, 300.0, "#ffd93d20", "#ffd93d", 1.5);
    svg.rect(445.0, 45.0, 260.0, 300.0, "#ffd93d20", "#ffd93d", 1.5);

    // Level 2: Quadrants
    svg.rect(48.0, 48.0, 160.0, 135.0, "#6bcb7720", "#6bcb77", 1.0);
    svg.rect(48.0, 195.0, 160.0, 135.0, "#6bcb7720", "#6bcb77", 1.0);
    svg.rect(448.0, 48.0, 165.0, 135.0, "#6bcb7720", "#6bcb77", 1.0);
    svg.rect(448.0, 195.0, 165.0, 135.0, "#6bcb7720", "#6bcb77", 1.0);

    // Draw actual AABBs (leaves)
    for aabb in &aabbs {
        svg.rect(
            aabb.min.x,
            aabb.min.y,
            aabb.max.x - aabb.min.x,
            aabb.max.y - aabb.min.y,
            "#00d4ff40",
            "#00d4ff",
            1.5,
        );
    }

    // Legend
    svg.rect(50.0, 365.0, 15.0, 15.0, "#ff6b6b20", "#ff6b6b", 1.0);
    svg.text(70.0, 377.0, "Root", 11.0, "#808080");

    svg.rect(130.0, 365.0, 15.0, 15.0, "#ffd93d20", "#ffd93d", 1.0);
    svg.text(150.0, 377.0, "Level 1", 11.0, "#808080");

    svg.rect(220.0, 365.0, 15.0, 15.0, "#6bcb7720", "#6bcb77", 1.0);
    svg.text(240.0, 377.0, "Level 2", 11.0, "#808080");

    svg.rect(320.0, 365.0, 15.0, 15.0, "#00d4ff40", "#00d4ff", 1.0);
    svg.text(340.0, 377.0, "Leaf AABBs", 11.0, "#808080");

    svg.text(500.0, 377.0, "Spatial partitioning enables O(log n) queries", 11.0, "#606060");

    svg.save("screenshots/bvh.svg");
    println!("Generated bvh.svg");
}

fn generate_sdf() {
    let mut svg = Svg::new(WIDTH, HEIGHT);

    svg.text(320.0, 30.0, "Signed Distance Field", 16.0, "#e0e0e0");

    // Create a star polygon
    let center = Point2::new(200.0, 200.0);
    let outer_r = 120.0;
    let inner_r = 50.0;
    let points_count = 5;

    let mut star_points = Vec::new();
    for i in 0..(points_count * 2) {
        let angle = (i as f64) * PI / points_count as f64 - PI / 2.0;
        let r = if i % 2 == 0 { outer_r } else { inner_r };
        star_points.push(Point2::new(center.x + r * angle.cos(), center.y + r * angle.sin()));
    }

    // Draw SDF as colored grid
    let grid_size = 10.0;

    // Sample the SDF and draw colored squares
    for gx in 0..40 {
        for gy in 0..38 {
            let x = gx as f64 * grid_size + 10.0;
            let y = gy as f64 * grid_size + 10.0;
            let p = Point2::new(x, y);

            let dist = sdf_polygon(p, &star_points);

            // Color based on distance
            let (r, g, b) = if dist < 0.0 {
                // Inside: blue to purple
                let t = (dist.abs() / 80.0).min(1.0);
                (
                    (100.0 + t * 100.0) as u8,
                    (50.0 + t * 50.0) as u8,
                    (200.0 - t * 50.0) as u8,
                )
            } else {
                // Outside: orange to dark
                let t = (dist / 100.0).min(1.0);
                (
                    (255.0 - t * 200.0) as u8,
                    (150.0 - t * 130.0) as u8,
                    (50.0 - t * 30.0) as u8,
                )
            };

            let color = format!("#{:02x}{:02x}{:02x}", r, g, b);
            svg.rect(x, y, grid_size, grid_size, &color, "none", 0.0);
        }
    }

    // Draw the star polygon outline
    svg.polygon(&star_points, "none", "#ffffff", 2.0);

    // Right side: circle SDF
    let circle = Circle::new(Point2::new(600.0, 200.0), 80.0);

    for gx in 0..40 {
        for gy in 0..38 {
            let x = gx as f64 * grid_size + 410.0;
            let y = gy as f64 * grid_size + 10.0;
            let p = Point2::new(x, y);

            let dist = circle.signed_distance(p);

            let (r, g, b) = if dist < 0.0 {
                let t = (dist.abs() / 80.0).min(1.0);
                (
                    (50.0 + t * 50.0) as u8,
                    (150.0 + t * 100.0) as u8,
                    (100.0 + t * 100.0) as u8,
                )
            } else {
                let t = (dist / 100.0).min(1.0);
                (
                    (100.0 - t * 80.0) as u8,
                    (200.0 - t * 180.0) as u8,
                    (150.0 - t * 130.0) as u8,
                )
            };

            let color = format!("#{:02x}{:02x}{:02x}", r, g, b);
            svg.rect(x, y, grid_size, grid_size, &color, "none", 0.0);
        }
    }

    // Circle outline
    svg.circle(600.0, 200.0, 80.0, "none", "#ffffff", 2.0);

    // Labels
    svg.text(150.0, 380.0, "Star polygon SDF", 12.0, "#808080");
    svg.text(560.0, 380.0, "Circle SDF", 12.0, "#808080");

    svg.save("screenshots/sdf.svg");
    println!("Generated sdf.svg");
}
