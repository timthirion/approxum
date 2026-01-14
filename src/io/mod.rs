//! Input/output utilities for geometric data.
//!
//! Provides parsing and serialization for common formats like SVG paths.

mod svg;

pub use svg::{
    parse_svg_path, polygon_to_svg_path, polyline_to_svg_path, svg_path_to_polylines, SvgCommand,
    SvgParseError, SvgPath,
};
