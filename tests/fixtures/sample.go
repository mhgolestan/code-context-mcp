package sample

import "fmt"

// Greet returns a greeting. Line numbers are asserted in test_parser.py.
func Greet(name string) string {
	return fmt.Sprintf("hello %s", name)
}

type Counter struct {
	n int
}

func (c *Counter) Bump() int {
	c.n++
	return c.n
}
