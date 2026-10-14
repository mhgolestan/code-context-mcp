require_relative "lexer"
require "json"

module Acme
  # Parses text into tokens.
  # Second line of the comment.
  class Parser
    def parse(text)
      tokens = lex(text)
      self.build(tokens).finish
    end

    def self.create
      new
    end

    private

    def lex(text)
      Lexer.new(text).split
    end
  end
end
