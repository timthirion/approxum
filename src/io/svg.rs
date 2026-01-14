//! SVG path parsing and export.
//!
//! Parses SVG path data (the `d` attribute) and converts to/from polylines.
//!
//! # Supported Commands
//!
//! - `M`/`m` - Move to (absolute/relative)
//! - `L`/`l` - Line to
//! - `H`/`h` - Horizontal line to
//! - `V`/`v` - Vertical line to
//! - `C`/`c` - Cubic Bézier curve
//! - `S`/`s` - Smooth cubic Bézier
//! - `Q`/`q` - Quadratic Bézier curve
//! - `T`/`t` - Smooth quadratic Bézier
//! - `A`/`a` - Elliptical arc
//! - `Z`/`z` - Close path
//!
//! # Example
//!
//! ```
//! use approxum::io::{parse_svg_path, svg_path_to_polylines, polyline_to_svg_path};
//! use approxum::Point2;
//!
//! // Parse an SVG path
//! let path = parse_svg_path("M 0 0 L 10 0 L 10 10 Z").unwrap();
//!
//! // Convert to polylines
//! let polylines = svg_path_to_polylines(&path, 0.1);
//!
//! // Export back to SVG
//! let svg = polyline_to_svg_path(&polylines[0], true);
//! ```

use crate::curves::{CubicBezier2, QuadraticBezier2};
use crate::polygon::Polygon;
use crate::primitives::Point2;
use num_traits::Float;
use std::fmt;
use std::str::FromStr;

/// Error type for SVG path parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum SvgParseError {
    /// Unexpected character encountered.
    UnexpectedChar(char, usize),
    /// Expected a number but found something else.
    ExpectedNumber(usize),
    /// Invalid number format.
    InvalidNumber(String, usize),
    /// Unexpected end of input.
    UnexpectedEnd,
    /// Unknown command character.
    UnknownCommand(char, usize),
    /// Invalid arc parameters.
    InvalidArcParams(usize),
}

impl fmt::Display for SvgParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SvgParseError::UnexpectedChar(c, pos) => {
                write!(f, "Unexpected character '{}' at position {}", c, pos)
            }
            SvgParseError::ExpectedNumber(pos) => {
                write!(f, "Expected number at position {}", pos)
            }
            SvgParseError::InvalidNumber(s, pos) => {
                write!(f, "Invalid number '{}' at position {}", s, pos)
            }
            SvgParseError::UnexpectedEnd => write!(f, "Unexpected end of path"),
            SvgParseError::UnknownCommand(c, pos) => {
                write!(f, "Unknown command '{}' at position {}", c, pos)
            }
            SvgParseError::InvalidArcParams(pos) => {
                write!(f, "Invalid arc parameters at position {}", pos)
            }
        }
    }
}

impl std::error::Error for SvgParseError {}

/// An SVG path command.
#[derive(Debug, Clone, PartialEq)]
pub enum SvgCommand<F> {
    /// Move to (absolute).
    MoveTo(Point2<F>),
    /// Move to (relative).
    MoveToRel(Point2<F>),
    /// Line to (absolute).
    LineTo(Point2<F>),
    /// Line to (relative).
    LineToRel(Point2<F>),
    /// Horizontal line to (absolute).
    HorizontalTo(F),
    /// Horizontal line to (relative).
    HorizontalToRel(F),
    /// Vertical line to (absolute).
    VerticalTo(F),
    /// Vertical line to (relative).
    VerticalToRel(F),
    /// Cubic Bézier curve (absolute).
    CubicTo(Point2<F>, Point2<F>, Point2<F>),
    /// Cubic Bézier curve (relative).
    CubicToRel(Point2<F>, Point2<F>, Point2<F>),
    /// Smooth cubic Bézier (absolute).
    SmoothCubicTo(Point2<F>, Point2<F>),
    /// Smooth cubic Bézier (relative).
    SmoothCubicToRel(Point2<F>, Point2<F>),
    /// Quadratic Bézier curve (absolute).
    QuadraticTo(Point2<F>, Point2<F>),
    /// Quadratic Bézier curve (relative).
    QuadraticToRel(Point2<F>, Point2<F>),
    /// Smooth quadratic Bézier (absolute).
    SmoothQuadraticTo(Point2<F>),
    /// Smooth quadratic Bézier (relative).
    SmoothQuadraticToRel(Point2<F>),
    /// Elliptical arc (absolute).
    ArcTo {
        rx: F,
        ry: F,
        x_axis_rotation: F,
        large_arc: bool,
        sweep: bool,
        end: Point2<F>,
    },
    /// Elliptical arc (relative).
    ArcToRel {
        rx: F,
        ry: F,
        x_axis_rotation: F,
        large_arc: bool,
        sweep: bool,
        end: Point2<F>,
    },
    /// Close path.
    ClosePath,
}

/// A parsed SVG path consisting of multiple commands.
#[derive(Debug, Clone, Default)]
pub struct SvgPath<F> {
    /// The list of path commands.
    pub commands: Vec<SvgCommand<F>>,
}

impl<F: Float> SvgPath<F> {
    /// Creates an empty path.
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
        }
    }

    /// Returns true if the path has no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Returns the number of commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }
}

/// Parses an SVG path string into a list of commands.
///
/// # Example
///
/// ```
/// use approxum::io::parse_svg_path;
///
/// let path = parse_svg_path::<f64>("M 0 0 L 10 0 L 10 10 Z").unwrap();
/// assert_eq!(path.len(), 4);
/// ```
pub fn parse_svg_path<F: Float + FromStr>(s: &str) -> Result<SvgPath<F>, SvgParseError> {
    let mut parser = SvgPathParser::new(s);
    parser.parse()
}

/// Converts an SVG path to a list of polylines.
///
/// Curves are flattened to line segments using the given tolerance.
/// Each subpath (starting with M/m) becomes a separate polyline.
/// Closed paths have their first point duplicated at the end.
///
/// # Arguments
///
/// * `path` - The parsed SVG path
/// * `tolerance` - Maximum deviation for curve flattening
///
/// # Returns
///
/// A vector of polylines, one per subpath.
///
/// # Example
///
/// ```
/// use approxum::io::{parse_svg_path, svg_path_to_polylines};
///
/// let path = parse_svg_path::<f64>("M 0 0 L 10 0 L 10 10 Z").unwrap();
/// let polylines = svg_path_to_polylines(&path, 0.1);
/// assert_eq!(polylines.len(), 1);
/// assert_eq!(polylines[0].len(), 4); // 3 points + closing point
/// ```
pub fn svg_path_to_polylines<F: Float>(path: &SvgPath<F>, tolerance: F) -> Vec<Vec<Point2<F>>> {
    let mut result: Vec<Vec<Point2<F>>> = Vec::new();
    let mut current_polyline: Vec<Point2<F>> = Vec::new();
    let mut current_pos = Point2::new(F::zero(), F::zero());
    let mut subpath_start = current_pos;
    let mut last_control: Option<Point2<F>> = None;
    let mut last_command_was_cubic = false;
    let mut last_command_was_quadratic = false;

    for cmd in &path.commands {
        match cmd {
            SvgCommand::MoveTo(p) => {
                if !current_polyline.is_empty() {
                    result.push(current_polyline);
                    current_polyline = Vec::new();
                }
                current_pos = *p;
                subpath_start = current_pos;
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::MoveToRel(p) => {
                if !current_polyline.is_empty() {
                    result.push(current_polyline);
                    current_polyline = Vec::new();
                }
                current_pos = Point2::new(current_pos.x + p.x, current_pos.y + p.y);
                subpath_start = current_pos;
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::LineTo(p) => {
                current_pos = *p;
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::LineToRel(p) => {
                current_pos = Point2::new(current_pos.x + p.x, current_pos.y + p.y);
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::HorizontalTo(x) => {
                current_pos = Point2::new(*x, current_pos.y);
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::HorizontalToRel(dx) => {
                current_pos = Point2::new(current_pos.x + *dx, current_pos.y);
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::VerticalTo(y) => {
                current_pos = Point2::new(current_pos.x, *y);
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::VerticalToRel(dy) => {
                current_pos = Point2::new(current_pos.x, current_pos.y + *dy);
                current_polyline.push(current_pos);
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::CubicTo(c1, c2, end) => {
                let curve = CubicBezier2::new(current_pos, *c1, *c2, *end);
                let points = curve.to_polyline(tolerance);
                // Skip first point as it's the current position
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = *end;
                last_control = Some(*c2);
                last_command_was_cubic = true;
                last_command_was_quadratic = false;
            }
            SvgCommand::CubicToRel(c1, c2, end) => {
                let abs_c1 = Point2::new(current_pos.x + c1.x, current_pos.y + c1.y);
                let abs_c2 = Point2::new(current_pos.x + c2.x, current_pos.y + c2.y);
                let abs_end = Point2::new(current_pos.x + end.x, current_pos.y + end.y);
                let curve = CubicBezier2::new(current_pos, abs_c1, abs_c2, abs_end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = abs_end;
                last_control = Some(abs_c2);
                last_command_was_cubic = true;
                last_command_was_quadratic = false;
            }
            SvgCommand::SmoothCubicTo(c2, end) => {
                let c1 = if last_command_was_cubic {
                    if let Some(lc) = last_control {
                        // Reflect last control point
                        Point2::new(
                            current_pos.x + current_pos.x - lc.x,
                            current_pos.y + current_pos.y - lc.y,
                        )
                    } else {
                        current_pos
                    }
                } else {
                    current_pos
                };
                let curve = CubicBezier2::new(current_pos, c1, *c2, *end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = *end;
                last_control = Some(*c2);
                last_command_was_cubic = true;
                last_command_was_quadratic = false;
            }
            SvgCommand::SmoothCubicToRel(c2, end) => {
                let c1 = if last_command_was_cubic {
                    if let Some(lc) = last_control {
                        Point2::new(
                            current_pos.x + current_pos.x - lc.x,
                            current_pos.y + current_pos.y - lc.y,
                        )
                    } else {
                        current_pos
                    }
                } else {
                    current_pos
                };
                let abs_c2 = Point2::new(current_pos.x + c2.x, current_pos.y + c2.y);
                let abs_end = Point2::new(current_pos.x + end.x, current_pos.y + end.y);
                let curve = CubicBezier2::new(current_pos, c1, abs_c2, abs_end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = abs_end;
                last_control = Some(abs_c2);
                last_command_was_cubic = true;
                last_command_was_quadratic = false;
            }
            SvgCommand::QuadraticTo(c, end) => {
                let curve = QuadraticBezier2::new(current_pos, *c, *end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = *end;
                last_control = Some(*c);
                last_command_was_cubic = false;
                last_command_was_quadratic = true;
            }
            SvgCommand::QuadraticToRel(c, end) => {
                let abs_c = Point2::new(current_pos.x + c.x, current_pos.y + c.y);
                let abs_end = Point2::new(current_pos.x + end.x, current_pos.y + end.y);
                let curve = QuadraticBezier2::new(current_pos, abs_c, abs_end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = abs_end;
                last_control = Some(abs_c);
                last_command_was_cubic = false;
                last_command_was_quadratic = true;
            }
            SvgCommand::SmoothQuadraticTo(end) => {
                let c = if last_command_was_quadratic {
                    if let Some(lc) = last_control {
                        Point2::new(
                            current_pos.x + current_pos.x - lc.x,
                            current_pos.y + current_pos.y - lc.y,
                        )
                    } else {
                        current_pos
                    }
                } else {
                    current_pos
                };
                let curve = QuadraticBezier2::new(current_pos, c, *end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = *end;
                last_control = Some(c);
                last_command_was_cubic = false;
                last_command_was_quadratic = true;
            }
            SvgCommand::SmoothQuadraticToRel(end) => {
                let c = if last_command_was_quadratic {
                    if let Some(lc) = last_control {
                        Point2::new(
                            current_pos.x + current_pos.x - lc.x,
                            current_pos.y + current_pos.y - lc.y,
                        )
                    } else {
                        current_pos
                    }
                } else {
                    current_pos
                };
                let abs_end = Point2::new(current_pos.x + end.x, current_pos.y + end.y);
                let curve = QuadraticBezier2::new(current_pos, c, abs_end);
                let points = curve.to_polyline(tolerance);
                current_polyline.extend(points.into_iter().skip(1));
                current_pos = abs_end;
                last_control = Some(c);
                last_command_was_cubic = false;
                last_command_was_quadratic = true;
            }
            SvgCommand::ArcTo {
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                end,
            } => {
                let arc_points = arc_to_polyline(
                    current_pos,
                    *rx,
                    *ry,
                    *x_axis_rotation,
                    *large_arc,
                    *sweep,
                    *end,
                    tolerance,
                );
                current_polyline.extend(arc_points.into_iter().skip(1));
                current_pos = *end;
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::ArcToRel {
                rx,
                ry,
                x_axis_rotation,
                large_arc,
                sweep,
                end,
            } => {
                let abs_end = Point2::new(current_pos.x + end.x, current_pos.y + end.y);
                let arc_points = arc_to_polyline(
                    current_pos,
                    *rx,
                    *ry,
                    *x_axis_rotation,
                    *large_arc,
                    *sweep,
                    abs_end,
                    tolerance,
                );
                current_polyline.extend(arc_points.into_iter().skip(1));
                current_pos = abs_end;
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
            SvgCommand::ClosePath => {
                if current_pos.distance(subpath_start) > F::epsilon() {
                    current_polyline.push(subpath_start);
                }
                current_pos = subpath_start;
                last_control = None;
                last_command_was_cubic = false;
                last_command_was_quadratic = false;
            }
        }
    }

    if !current_polyline.is_empty() {
        result.push(current_polyline);
    }

    result
}

/// Converts a polyline to an SVG path string.
///
/// # Arguments
///
/// * `points` - The polyline vertices
/// * `closed` - Whether to close the path with 'Z'
///
/// # Returns
///
/// An SVG path string using M and L commands.
///
/// # Example
///
/// ```
/// use approxum::{Point2, io::polyline_to_svg_path};
///
/// let points = vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(10.0, 0.0),
///     Point2::new(10.0, 10.0),
/// ];
///
/// let svg = polyline_to_svg_path(&points, true);
/// assert!(svg.starts_with("M"));
/// assert!(svg.ends_with("Z"));
/// ```
pub fn polyline_to_svg_path<F: Float + fmt::Display>(points: &[Point2<F>], closed: bool) -> String {
    if points.is_empty() {
        return String::new();
    }

    let mut result = String::new();

    // Move to first point
    result.push_str(&format!("M {} {}", points[0].x, points[0].y));

    // Line to remaining points
    for p in &points[1..] {
        result.push_str(&format!(" L {} {}", p.x, p.y));
    }

    if closed {
        result.push_str(" Z");
    }

    result
}

/// Converts a polygon to an SVG path string.
///
/// # Example
///
/// ```
/// use approxum::{Point2, polygon::Polygon, io::polygon_to_svg_path};
///
/// let poly = Polygon::new(vec![
///     Point2::new(0.0, 0.0),
///     Point2::new(10.0, 0.0),
///     Point2::new(10.0, 10.0),
/// ]);
///
/// let svg = polygon_to_svg_path(&poly);
/// assert!(svg.ends_with("Z"));
/// ```
pub fn polygon_to_svg_path<F: Float + fmt::Display>(polygon: &Polygon<F>) -> String {
    polyline_to_svg_path(&polygon.vertices, true)
}

// ============================================================================
// Internal implementation
// ============================================================================

/// SVG path parser.
struct SvgPathParser<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    current_pos: usize,
}

impl<'a> SvgPathParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.char_indices().peekable(),
            current_pos: 0,
        }
    }

    fn parse<F: Float + FromStr>(&mut self) -> Result<SvgPath<F>, SvgParseError> {
        let mut path = SvgPath::new();

        while let Some(&(pos, c)) = self.chars.peek() {
            self.current_pos = pos;

            if c.is_whitespace() || c == ',' {
                self.chars.next();
                continue;
            }

            if c.is_alphabetic() {
                let cmd = self.parse_command()?;
                path.commands.extend(cmd);
            } else {
                return Err(SvgParseError::UnexpectedChar(c, pos));
            }
        }

        Ok(path)
    }

    fn parse_command<F: Float + FromStr>(&mut self) -> Result<Vec<SvgCommand<F>>, SvgParseError> {
        let (pos, cmd_char) = self.chars.next().ok_or(SvgParseError::UnexpectedEnd)?;
        self.current_pos = pos;

        let mut commands = Vec::new();

        match cmd_char {
            'M' => {
                let (x, y) = self.parse_coordinate_pair()?;
                commands.push(SvgCommand::MoveTo(Point2::new(x, y)));
                // Subsequent pairs are implicit LineTo
                while self.has_number() {
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::LineTo(Point2::new(x, y)));
                }
            }
            'm' => {
                let (x, y) = self.parse_coordinate_pair()?;
                commands.push(SvgCommand::MoveToRel(Point2::new(x, y)));
                while self.has_number() {
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::LineToRel(Point2::new(x, y)));
                }
            }
            'L' => {
                loop {
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::LineTo(Point2::new(x, y)));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'l' => {
                loop {
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::LineToRel(Point2::new(x, y)));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'H' => {
                loop {
                    let x = self.parse_number()?;
                    commands.push(SvgCommand::HorizontalTo(x));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'h' => {
                loop {
                    let x = self.parse_number()?;
                    commands.push(SvgCommand::HorizontalToRel(x));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'V' => {
                loop {
                    let y = self.parse_number()?;
                    commands.push(SvgCommand::VerticalTo(y));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'v' => {
                loop {
                    let y = self.parse_number()?;
                    commands.push(SvgCommand::VerticalToRel(y));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'C' => {
                loop {
                    let (x1, y1) = self.parse_coordinate_pair()?;
                    let (x2, y2) = self.parse_coordinate_pair()?;
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::CubicTo(
                        Point2::new(x1, y1),
                        Point2::new(x2, y2),
                        Point2::new(x, y),
                    ));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'c' => {
                loop {
                    let (x1, y1) = self.parse_coordinate_pair()?;
                    let (x2, y2) = self.parse_coordinate_pair()?;
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::CubicToRel(
                        Point2::new(x1, y1),
                        Point2::new(x2, y2),
                        Point2::new(x, y),
                    ));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'S' => {
                loop {
                    let (x2, y2) = self.parse_coordinate_pair()?;
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::SmoothCubicTo(
                        Point2::new(x2, y2),
                        Point2::new(x, y),
                    ));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            's' => {
                loop {
                    let (x2, y2) = self.parse_coordinate_pair()?;
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::SmoothCubicToRel(
                        Point2::new(x2, y2),
                        Point2::new(x, y),
                    ));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'Q' => {
                loop {
                    let (x1, y1) = self.parse_coordinate_pair()?;
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::QuadraticTo(
                        Point2::new(x1, y1),
                        Point2::new(x, y),
                    ));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'q' => {
                loop {
                    let (x1, y1) = self.parse_coordinate_pair()?;
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::QuadraticToRel(
                        Point2::new(x1, y1),
                        Point2::new(x, y),
                    ));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'T' => {
                loop {
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::SmoothQuadraticTo(Point2::new(x, y)));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            't' => {
                loop {
                    let (x, y) = self.parse_coordinate_pair()?;
                    commands.push(SvgCommand::SmoothQuadraticToRel(Point2::new(x, y)));
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'A' => {
                loop {
                    let arc = self.parse_arc_params(false)?;
                    commands.push(arc);
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'a' => {
                loop {
                    let arc = self.parse_arc_params(true)?;
                    commands.push(arc);
                    if !self.has_number() {
                        break;
                    }
                }
            }
            'Z' | 'z' => {
                commands.push(SvgCommand::ClosePath);
            }
            _ => {
                return Err(SvgParseError::UnknownCommand(cmd_char, pos));
            }
        }

        Ok(commands)
    }

    fn skip_whitespace_and_commas(&mut self) {
        while let Some(&(_, c)) = self.chars.peek() {
            if c.is_whitespace() || c == ',' {
                self.chars.next();
            } else {
                break;
            }
        }
    }

    fn has_number(&mut self) -> bool {
        self.skip_whitespace_and_commas();
        if let Some(&(_, c)) = self.chars.peek() {
            c.is_ascii_digit() || c == '-' || c == '+' || c == '.'
        } else {
            false
        }
    }

    fn parse_number<F: Float + FromStr>(&mut self) -> Result<F, SvgParseError> {
        self.skip_whitespace_and_commas();

        let start = self.chars.peek().map(|&(i, _)| i).unwrap_or(self.input.len());
        let mut end = start;

        // Optional sign
        if let Some(&(_, c)) = self.chars.peek() {
            if c == '-' || c == '+' {
                self.chars.next();
            }
        }

        // Integer part
        while let Some(&(i, c)) = self.chars.peek() {
            if c.is_ascii_digit() {
                end = i + 1;
                self.chars.next();
            } else {
                break;
            }
        }

        // Decimal part
        if let Some(&(_, '.')) = self.chars.peek() {
            self.chars.next();
            while let Some(&(i, c)) = self.chars.peek() {
                if c.is_ascii_digit() {
                    end = i + 1;
                    self.chars.next();
                } else {
                    break;
                }
            }
        }

        // Exponent part
        if let Some(&(_, c)) = self.chars.peek() {
            if c == 'e' || c == 'E' {
                self.chars.next();
                if let Some(&(_, c)) = self.chars.peek() {
                    if c == '-' || c == '+' {
                        self.chars.next();
                    }
                }
                while let Some(&(i, c)) = self.chars.peek() {
                    if c.is_ascii_digit() {
                        end = i + 1;
                        self.chars.next();
                    } else {
                        break;
                    }
                }
            }
        }

        if end == start {
            return Err(SvgParseError::ExpectedNumber(start));
        }

        let num_str = &self.input[start..end];
        num_str
            .parse()
            .map_err(|_| SvgParseError::InvalidNumber(num_str.to_string(), start))
    }

    fn parse_coordinate_pair<F: Float + FromStr>(&mut self) -> Result<(F, F), SvgParseError> {
        let x = self.parse_number()?;
        let y = self.parse_number()?;
        Ok((x, y))
    }

    fn parse_arc_params<F: Float + FromStr>(
        &mut self,
        relative: bool,
    ) -> Result<SvgCommand<F>, SvgParseError> {
        let rx = self.parse_number()?;
        let ry = self.parse_number()?;
        let x_axis_rotation = self.parse_number()?;

        self.skip_whitespace_and_commas();
        let large_arc_flag = self.parse_flag()?;

        self.skip_whitespace_and_commas();
        let sweep_flag = self.parse_flag()?;

        let (x, y) = self.parse_coordinate_pair()?;

        if relative {
            Ok(SvgCommand::ArcToRel {
                rx,
                ry,
                x_axis_rotation,
                large_arc: large_arc_flag,
                sweep: sweep_flag,
                end: Point2::new(x, y),
            })
        } else {
            Ok(SvgCommand::ArcTo {
                rx,
                ry,
                x_axis_rotation,
                large_arc: large_arc_flag,
                sweep: sweep_flag,
                end: Point2::new(x, y),
            })
        }
    }

    fn parse_flag(&mut self) -> Result<bool, SvgParseError> {
        self.skip_whitespace_and_commas();
        match self.chars.next() {
            Some((_, '0')) => Ok(false),
            Some((_, '1')) => Ok(true),
            Some((pos, c)) => Err(SvgParseError::UnexpectedChar(c, pos)),
            None => Err(SvgParseError::UnexpectedEnd),
        }
    }
}

/// Converts an SVG arc to a polyline.
fn arc_to_polyline<F: Float>(
    start: Point2<F>,
    rx: F,
    ry: F,
    x_axis_rotation: F,
    large_arc: bool,
    sweep: bool,
    end: Point2<F>,
    tolerance: F,
) -> Vec<Point2<F>> {
    // Handle degenerate cases
    if rx.abs() < F::epsilon() || ry.abs() < F::epsilon() {
        return vec![start, end];
    }

    if start.distance(end) < F::epsilon() {
        return vec![start];
    }

    // Convert to center parameterization
    let phi = x_axis_rotation * F::from(std::f64::consts::PI / 180.0).unwrap();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // Step 1: Compute (x1', y1')
    let dx = (start.x - end.x) / (F::one() + F::one());
    let dy = (start.y - end.y) / (F::one() + F::one());
    let x1_prime = cos_phi * dx + sin_phi * dy;
    let y1_prime = -sin_phi * dx + cos_phi * dy;

    // Ensure radii are large enough
    let mut rx = rx.abs();
    let mut ry = ry.abs();

    let lambda = (x1_prime * x1_prime) / (rx * rx) + (y1_prime * y1_prime) / (ry * ry);
    if lambda > F::one() {
        let sqrt_lambda = lambda.sqrt();
        rx = sqrt_lambda * rx;
        ry = sqrt_lambda * ry;
    }

    // Step 2: Compute (cx', cy')
    let rx_sq = rx * rx;
    let ry_sq = ry * ry;
    let x1_prime_sq = x1_prime * x1_prime;
    let y1_prime_sq = y1_prime * y1_prime;

    let mut sq = (rx_sq * ry_sq - rx_sq * y1_prime_sq - ry_sq * x1_prime_sq)
        / (rx_sq * y1_prime_sq + ry_sq * x1_prime_sq);

    if sq < F::zero() {
        sq = F::zero();
    }

    let coef = if large_arc == sweep {
        -sq.sqrt()
    } else {
        sq.sqrt()
    };

    let cx_prime = coef * rx * y1_prime / ry;
    let cy_prime = -coef * ry * x1_prime / rx;

    // Step 3: Compute (cx, cy)
    let mid_x = (start.x + end.x) / (F::one() + F::one());
    let mid_y = (start.y + end.y) / (F::one() + F::one());
    let cx = cos_phi * cx_prime - sin_phi * cy_prime + mid_x;
    let cy = sin_phi * cx_prime + cos_phi * cy_prime + mid_y;

    // Step 4: Compute angles
    let ux = (x1_prime - cx_prime) / rx;
    let uy = (y1_prime - cy_prime) / ry;
    let vx = (-x1_prime - cx_prime) / rx;
    let vy = (-y1_prime - cy_prime) / ry;

    let two = F::one() + F::one();
    let pi = F::from(std::f64::consts::PI).unwrap();

    let n = (ux * ux + uy * uy).sqrt();
    let theta1 = if uy >= F::zero() {
        (ux / n).acos()
    } else {
        -(ux / n).acos()
    };

    let n2 = ((ux * ux + uy * uy) * (vx * vx + vy * vy)).sqrt();
    let dot = ux * vx + uy * vy;
    let mut dtheta = (dot / n2).acos();

    if ux * vy - uy * vx < F::zero() {
        dtheta = -dtheta;
    }

    if sweep && dtheta < F::zero() {
        dtheta = dtheta + two * pi;
    } else if !sweep && dtheta > F::zero() {
        dtheta = dtheta - two * pi;
    }

    // Generate points
    let num_segments = ((dtheta.abs() / (tolerance / rx.min(ry)).acos().max(F::from(0.1).unwrap()))
        .ceil()
        .max(F::one()))
    .to_usize()
    .unwrap_or(8)
    .min(360);

    let mut points = Vec::with_capacity(num_segments + 1);
    points.push(start);

    for i in 1..num_segments {
        let t = F::from(i).unwrap() / F::from(num_segments).unwrap();
        let angle = theta1 + t * dtheta;

        let x_prime = rx * angle.cos();
        let y_prime = ry * angle.sin();

        let x = cos_phi * x_prime - sin_phi * y_prime + cx;
        let y = sin_phi * x_prime + cos_phi * y_prime + cy;

        points.push(Point2::new(x, y));
    }

    points.push(end);
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parse_moveto_lineto() {
        let path = parse_svg_path::<f64>("M 0 0 L 10 0 L 10 10").unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_parse_closed_path() {
        let path = parse_svg_path::<f64>("M 0 0 L 10 0 L 10 10 Z").unwrap();
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_parse_relative_commands() {
        let path = parse_svg_path::<f64>("m 0 0 l 10 0 l 0 10 z").unwrap();
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_parse_horizontal_vertical() {
        let path = parse_svg_path::<f64>("M 0 0 H 10 V 10 H 0 Z").unwrap();
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn test_parse_cubic_bezier() {
        let path = parse_svg_path::<f64>("M 0 0 C 1 2 3 2 4 0").unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_parse_quadratic_bezier() {
        let path = parse_svg_path::<f64>("M 0 0 Q 2 4 4 0").unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_parse_smooth_curves() {
        let path = parse_svg_path::<f64>("M 0 0 C 1 2 3 2 4 0 S 7 2 8 0").unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_parse_arc() {
        let path = parse_svg_path::<f64>("M 0 0 A 5 5 0 0 1 10 0").unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_parse_implicit_lineto() {
        // After M, additional coordinate pairs are treated as L
        let path = parse_svg_path::<f64>("M 0 0 10 0 10 10").unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_parse_no_spaces() {
        let path = parse_svg_path::<f64>("M0,0L10,0L10,10Z").unwrap();
        assert_eq!(path.len(), 4);
    }

    #[test]
    fn test_parse_negative_numbers() {
        let path = parse_svg_path::<f64>("M -5 -5 L 10 -10 L -10 10").unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_parse_decimal_numbers() {
        let path = parse_svg_path::<f64>("M 0.5 0.5 L 10.25 0.75").unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_parse_scientific_notation() {
        let path = parse_svg_path::<f64>("M 1e2 2e-1 L 1.5e1 2.5e0").unwrap();
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_to_polylines_simple() {
        let path = parse_svg_path::<f64>("M 0 0 L 10 0 L 10 10 Z").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);

        assert_eq!(polylines.len(), 1);
        assert_eq!(polylines[0].len(), 4);
        assert_relative_eq!(polylines[0][0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polylines[0][3].x, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_polylines_cubic() {
        let path = parse_svg_path::<f64>("M 0 0 C 1 2 3 2 4 0").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);

        assert_eq!(polylines.len(), 1);
        assert!(polylines[0].len() >= 2);
        assert_relative_eq!(polylines[0][0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polylines[0].last().unwrap().x, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_polylines_multiple_subpaths() {
        let path = parse_svg_path::<f64>("M 0 0 L 10 0 M 20 0 L 30 0").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);

        assert_eq!(polylines.len(), 2);
    }

    #[test]
    fn test_polyline_to_svg() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ];

        let svg = polyline_to_svg_path(&points, false);
        assert!(svg.starts_with("M 0 0"));
        assert!(svg.contains("L 10 0"));
        assert!(!svg.ends_with("Z"));
    }

    #[test]
    fn test_polyline_to_svg_closed() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ];

        let svg = polyline_to_svg_path(&points, true);
        assert!(svg.ends_with("Z"));
    }

    #[test]
    fn test_polygon_to_svg() {
        let poly = Polygon::new(vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
        ]);

        let svg = polygon_to_svg_path(&poly);
        assert!(svg.starts_with("M"));
        assert!(svg.ends_with("Z"));
    }

    #[test]
    fn test_roundtrip() {
        let original = "M 0 0 L 10 0 L 10 10 L 0 10 Z";
        let path = parse_svg_path::<f64>(original).unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);
        let exported = polyline_to_svg_path(&polylines[0], true);

        // Parse again and check
        let path2 = parse_svg_path::<f64>(&exported).unwrap();
        let polylines2 = svg_path_to_polylines(&path2, 0.1);

        assert_eq!(polylines[0].len(), polylines2[0].len());
    }

    #[test]
    fn test_parse_error_unknown_command() {
        let result = parse_svg_path::<f64>("M 0 0 X 10 10");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty() {
        let path = parse_svg_path::<f64>("").unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_arc_to_polylines() {
        let path = parse_svg_path::<f64>("M 0 0 A 10 10 0 0 1 20 0").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.5);

        assert_eq!(polylines.len(), 1);
        assert!(polylines[0].len() > 2);

        // Start and end should be correct
        assert_relative_eq!(polylines[0][0].x, 0.0, epsilon = 0.1);
        assert_relative_eq!(polylines[0].last().unwrap().x, 20.0, epsilon = 0.1);
    }

    #[test]
    fn test_f32_support() {
        let path = parse_svg_path::<f32>("M 0 0 L 10 0 L 10 10 Z").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);
        assert_eq!(polylines.len(), 1);
    }

    #[test]
    fn test_relative_cubic() {
        let path = parse_svg_path::<f64>("M 0 0 c 1 2 3 2 4 0").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);

        assert_relative_eq!(polylines[0][0].x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(polylines[0].last().unwrap().x, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_smooth_quadratic() {
        let path = parse_svg_path::<f64>("M 0 0 Q 2 4 4 0 T 8 0").unwrap();
        let polylines = svg_path_to_polylines(&path, 0.1);

        assert!(polylines[0].len() > 2);
        assert_relative_eq!(polylines[0].last().unwrap().x, 8.0, epsilon = 1e-10);
    }
}
