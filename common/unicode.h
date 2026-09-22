#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <string>

// UTF-8 parsing utilities for streaming-aware unicode support

struct utf8_parse_result {
    uint32_t codepoint;      // Decoded codepoint (only valid if status == SUCCESS)
    size_t bytes_consumed;   // How many bytes this codepoint uses (1-4)
    enum status { SUCCESS, INCOMPLETE, INVALID } status;

    utf8_parse_result(enum status s, uint32_t cp = 0, size_t bytes = 0)
        : codepoint(cp), bytes_consumed(bytes), status(s) {}
};

// Determine the expected length of a UTF-8 sequence from its first byte
// Returns 0 for invalid first bytes
size_t common_utf8_sequence_length(unsigned char first_byte);

// Check if a string ends with a complete UTF-8 sequence.
bool common_utf8_is_complete(const std::string & s);

// Parse a single UTF-8 codepoint from input
utf8_parse_result common_parse_utf8_codepoint(std::string_view input, size_t offset);

// Replace each malformed UTF-8 sequence in input with U+FFFD and store the
// result in output. A truncated sequence at the very end of input is copied
// as-is so that a partial input can still complete when more data arrives.
// Returns false (leaving output untouched) when input is already valid UTF-8.
bool common_utf8_sanitize(const std::string & input, std::string & output);

std::string common_unicode_cpts_to_utf8(const std::vector<uint32_t> & cps);
std::string common_unicode_cpt_to_utf8(uint32_t cpt);
