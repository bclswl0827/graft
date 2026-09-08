package main

import (
	"image"
	"math"
)

const foregroundDifference = 24 * 257

// detectTextLines segments high-contrast document images into lines and words.
// The returned rectangles retain the source image coordinate system.
func detectTextLines(source image.Image) [][]image.Rectangle {
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	background := borderColor(source)
	mask := make([]bool, width*height)
	rowCounts := make([]int, height)
	for y := range height {
		for x := range width {
			foreground := differsFrom(
				opaqueRGB(source, bounds.Min.X+x, bounds.Min.Y+y),
				background,
			)
			mask[y*width+x] = foreground
			if foreground {
				rowCounts[y]++
			}
		}
	}

	minimumRowPixels := max(2, width/1000)
	rowBands := occupiedBands(rowCounts, minimumRowPixels, 2)
	lines := make([][]image.Rectangle, 0, len(rowBands))
	for _, band := range rowBands {
		if band[1]-band[0] < 3 {
			continue
		}
		words := wordRectangles(mask, width, height, band, bounds.Min)
		if len(words) != 0 {
			lines = append(lines, words)
		}
	}
	if len(lines) == 0 {
		return [][]image.Rectangle{{bounds}}
	}
	return lines
}

func borderColor(source image.Image) [3]float64 {
	bounds := source.Bounds()
	points := [...]image.Point{
		bounds.Min,
		{X: bounds.Max.X - 1, Y: bounds.Min.Y},
		{X: bounds.Min.X, Y: bounds.Max.Y - 1},
		{X: bounds.Max.X - 1, Y: bounds.Max.Y - 1},
	}
	var result [3]float64
	for _, point := range points {
		color := opaqueRGB(source, point.X, point.Y)
		for channel := range result {
			result[channel] += color[channel] / float64(len(points))
		}
	}
	return result
}

func differsFrom(color, background [3]float64) bool {
	for channel := range color {
		if math.Abs(color[channel]-background[channel]) >= foregroundDifference {
			return true
		}
	}
	return false
}

func occupiedBands(counts []int, minimumCount, maximumGap int) [][2]int {
	var bands [][2]int
	start, lastOccupied := -1, -1
	for index, count := range counts {
		if count >= minimumCount {
			if start == -1 {
				start = index
			}
			lastOccupied = index
			continue
		}
		if start != -1 && index-lastOccupied > maximumGap {
			bands = append(bands, [2]int{start, lastOccupied + 1})
			start, lastOccupied = -1, -1
		}
	}
	if start != -1 {
		bands = append(bands, [2]int{start, lastOccupied + 1})
	}
	return bands
}

func wordRectangles(
	mask []bool,
	width, height int,
	rowBand [2]int,
	origin image.Point,
) []image.Rectangle {
	columnCounts := make([]int, width)
	for y := rowBand[0]; y < rowBand[1]; y++ {
		for x := range width {
			if mask[y*width+x] {
				columnCounts[x]++
			}
		}
	}

	minimumWordGap := max(2, (rowBand[1]-rowBand[0])/5)
	columnBands := occupiedBands(columnCounts, 1, minimumWordGap-1)
	words := make([]image.Rectangle, 0, len(columnBands))
	for _, band := range columnBands {
		if band[1]-band[0] < 2 {
			continue
		}
		rectangle := image.Rect(
			max(0, band[0]-2),
			max(0, rowBand[0]-2),
			min(width, band[1]+2),
			min(height, rowBand[1]+2),
		).Add(origin)
		words = append(words, rectangle)
	}
	return words
}
