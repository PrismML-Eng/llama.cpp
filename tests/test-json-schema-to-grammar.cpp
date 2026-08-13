#ifdef NDEBUG
#undef NDEBUG
#endif

#include "json-schema-to-grammar.h"

#include "../src/llama-grammar.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <fstream>
#include <sstream>
#include <regex>

static std::string trim(const std::string & source) {
    std::string s(source);
    s.erase(0,s.find_first_not_of(" \n\r\t"));
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    return std::regex_replace(s, std::regex("(^|\n)[ \t]+"), "$1");
}

enum TestCaseStatus {
    SUCCESS,
    FAILURE
};

struct TestCase {
    TestCaseStatus expected_status;
    std::string name;
    std::string schema;
    std::string expected_grammar;

    void _print_failure_header() const {
        fprintf(stderr, "#\n# Test '%s' failed.\n#\n%s\n", name.c_str(), schema.c_str());
    }
    void verify(const std::string & actual_grammar) const {
        if (trim(actual_grammar) != trim(expected_grammar)) {
        _print_failure_header();
        fprintf(stderr, "# EXPECTED:\n%s\n# ACTUAL:\n%s\n", expected_grammar.c_str(), actual_grammar.c_str());
        assert(false);
        }
    }
    void verify_expectation_parseable() const {
        try {
            llama_grammar_parser state;
            state.parse(expected_grammar.c_str());
            if (state.symbol_ids.find("root") == state.symbol_ids.end()) {
                throw std::runtime_error("Grammar failed to parse:\n" + expected_grammar);
            }
        } catch (const std::runtime_error & ex) {
            _print_failure_header();
            fprintf(stderr, "# GRAMMAR ERROR: %s\n", ex.what());
            assert(false);
        }
    }
    void verify_status(TestCaseStatus status) const {
        if (status != expected_status) {
            _print_failure_header();
            fprintf(stderr, "# EXPECTED STATUS: %s\n", expected_status == SUCCESS ? "SUCCESS" : "FAILURE");
            fprintf(stderr, "# ACTUAL STATUS: %s\n", status == SUCCESS ? "SUCCESS" : "FAILURE");
            assert(false);
        }
    }
};

static void write(const std::string & file, const std::string & content) {
    std::ofstream f;
    f.open(file.c_str());
    f << content.c_str();
    f.close();
}

static std::string read(const std::string & file) {
    std::ostringstream actuals;
    actuals << std::ifstream(file.c_str()).rdbuf();
    return actuals.str();
}

static void test_all(const std::string & lang, std::function<void(const TestCase &)> runner) {
    fprintf(stderr, "#\n# Testing JSON schema conversion (%s)\n#\n", lang.c_str());
    auto test = [&](const TestCase & tc) {
        fprintf(stderr, "- %s%s\n", tc.name.c_str(), tc.expected_status == FAILURE ? " (failure expected)" : "");
        runner(tc);
    };

    test({
        SUCCESS,
        "min 0",
        R"""({
            "type": "integer",
            "minimum": 0
        })""",
        R"""(
            root ::= ([0] | [1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 1",
        R"""({
            "type": "integer",
            "minimum": 1
        })""",
        R"""(
            root ::= ([1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 3",
        R"""({
            "type": "integer",
            "minimum": 3
        })""",
        R"""(
            root ::= ([1-2] [0-9]{1,15} | [3-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 9",
        R"""({
            "type": "integer",
            "minimum": 9
        })""",
        R"""(
            root ::= ([1-8] [0-9]{1,15} | [9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 10",
        R"""({
            "type": "integer",
            "minimum": 10
        })""",
        R"""(
            root ::= ([1] ([0-9]{1,15}) | [2-9] [0-9]{1,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 25",
        R"""({
            "type": "integer",
            "minimum": 25
        })""",
        R"""(
            root ::= ([1] [0-9]{2,15} | [2] ([0-4] [0-9]{1,14} | [5-9] [0-9]{0,14}) | [3-9] [0-9]{1,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 30",
        R"""({
            "type": "integer",
            "maximum": 30
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-2] [0-9] | [3] "0")) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -5",
        R"""({
            "type": "integer",
            "minimum": -5
        })""",
        R"""(
            root ::= ("-" ([0-5]) | [0] | [1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -123",
        R"""({
            "type": "integer",
            "minimum": -123
        })""",
        R"""(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0] | [1-9] [0-9]{0,15}) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max -5",
        R"""({
            "type": "integer",
            "maximum": -5
        })""",
        R"""(
            root ::= ("-" ([0-4] [0-9]{1,15} | [5-9] [0-9]{0,15})) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 1",
        R"""({
            "type": "integer",
            "maximum": 1
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-1]) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "max 100",
        R"""({
            "type": "integer",
            "maximum": 100
        })""",
        R"""(
            root ::= ("-" [1-9] [0-9]{0,15} | [0-9] | ([1-8] [0-9] | [9] [0-9]) | "100") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 0 max 23",
        R"""({
            "type": "integer",
            "minimum": 0,
            "maximum": 23
        })""",
        R"""(
            root ::= ([0-9] | ([1] [0-9] | [2] [0-3])) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 15 max 300",
        R"""({
            "type": "integer",
            "minimum": 15,
            "maximum": 300
        })""",
        R"""(
            root ::= (([1] ([5-9]) | [2-9] [0-9]) | ([1-2] [0-9]{2} | [3] "00")) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min 5 max 30",
        R"""({
            "type": "integer",
            "minimum": 5,
            "maximum": 30
        })""",
        R"""(
            root ::= ([5-9] | ([1-2] [0-9] | [3] "0")) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -123 max 42",
        R"""({
            "type": "integer",
            "minimum": -123,
            "maximum": 42
        })""",
        R"""(
            root ::= ("-" ([0-9] | ([1-8] [0-9] | [9] [0-9]) | "1" ([0-1] [0-9] | [2] [0-3])) | [0-9] | ([1-3] [0-9] | [4] [0-2])) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min -10 max 10",
        R"""({
            "type": "integer",
            "minimum": -10,
            "maximum": 10
        })""",
        R"""(
            root ::= ("-" ([0-9] | "10") | [0-9] | "10") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        FAILURE,
        "unknown type",
        R"""({
            "type": "kaboom"
        })""",
        ""
    });

    test({
        FAILURE,
        "invalid type",
        R"""({
            "type": 123
        })""",
        ""
    });

    test({
        SUCCESS,
        "empty schema (object)",
        "{}",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "exotic formats",
        R"""({
            "items": [
                { "format": "date" },
                { "format": "uuid" },
                { "format": "time" },
                { "format": "date-time" }
            ]
        })""",
        R"""(
            date ::= [0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )
            date-string ::= "\"" date "\"" space
            date-time ::= date "T" time
            date-time-string ::= "\"" date-time "\"" space
            root ::= "[" space tuple-0 "," space uuid "," space tuple-2 "," space tuple-3 "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            time ::= ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )
            time-string ::= "\"" time "\"" space
            tuple-0 ::= date-string
            tuple-2 ::= time-string
            tuple-3 ::= date-time-string
            uuid ::= "\"" [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space
        )"""
    });

    test({
        SUCCESS,
        "string",
        R"""({
            "type": "string"
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char* "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min length 1",
        R"""({
            "type": "string",
            "minLength": 1
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char+ "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min length 3",
        R"""({
            "type": "string",
            "minLength": 3
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{3,} "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ max length",
        R"""({
            "type": "string",
            "maxLength": 3
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{0,3} "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string w/ min & max length",
        R"""({
            "type": "string",
            "minLength": 1,
            "maxLength": 4
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "\"" char{1,4} "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "boolean",
        R"""({
            "type": "boolean"
        })""",
        R"""(
            root ::= ("true" | "false") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "integer",
        R"""({
            "type": "integer"
        })""",
        R"""(
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string const",
        R"""({
            "const": "foo"
        })""",
        R"""(
            root ::= "\"foo\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "non-string const",
        R"""({
            "const": 123
        })""",
        R"""(
            root ::= "123" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "non-string enum",
        R"""({
            "enum": ["red", "amber", "green", null, 42, ["foo"]]
        })""",
        R"""(
            root ::= ("\"red\"" | "\"amber\"" | "\"green\"" | "null" | "42" | "[\"foo\"]") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "string array",
        R"""({
            "type": "array",
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space (string ("," space string)*)? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "nullable string array",
        R"""({
            "type": ["array", "null"],
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            alternative-0 ::= "[" space (string ("," space string)*)? "]" space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            null ::= "null" space
            root ::= alternative-0 | null
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "tuple1",
        R"""({
            "prefixItems": [{ "type": "string" }]
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "[" space string "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "tuple2",
        R"""({
            "prefixItems": [{ "type": "string" }, { "type": "number" }]
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "[" space string "," space number "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "array with empty items",
        R"""({
            "type": "array",
            "items": {}
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= object
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= "[" space (item ("," space item)*)? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "array with empty items and prefixItems",
        R"""({
            "type": "array",
            "items": {},
            "prefixItems": { "type": "string" }
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= object
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= "[" space (item ("," space item)*)? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "number",
        R"""({
            "type": "number"
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "minItems",
        R"""({
            "items": {
                "type": "boolean"
            },
            "minItems": 2
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space boolean ("," space boolean)+ "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 0",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 0
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space  "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 1",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 1
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space boolean? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "maxItems 2",
        R"""({
            "items": {
                "type": "boolean"
            },
            "maxItems": 2
        })""",
        R"""(
            boolean ::= ("true" | "false") space
            root ::= "[" space (boolean ("," space boolean)?)? "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + maxItems",
        R"""({
            "items": {
                "type": ["number", "integer"]
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            item ::= number | integer
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "[" space item ("," space item){2,4} "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + max items with min + max values across zero",
        R"""({
            "items": {
                "type": "integer",
                "minimum": -12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            item ::= ("-" ([0-9] | "1" [0-2]) | [0-9] | ([1-8] [0-9] | [9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7])) space
            root ::= "[" space item ("," space item){2,4} "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "min + max items with min + max values",
        R"""({
            "items": {
                "type": "integer",
                "minimum": 12,
                "maximum": 207
            },
            "minItems": 3,
            "maxItems": 5
        })""",
        R"""(
            item ::= (([1] ([2-9]) | [2-9] [0-9]) | ([1] [0-9]{2} | [2] "0" [0-7])) space
            root ::= "[" space item ("," space item){2,4} "]" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "simple regexp",
        R"""({
            "type": "string",
            "pattern": "^abc?d*efg+(hij)?kl$"
        })""",
        R"""(
            root ::= "\"" ("ab" "c"? "d"* "ef" "g"+ ("hij")? "kl") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp escapes",
        R"""({
            "type": "string",
            "pattern": "^\\[\\]\\{\\}\\(\\)\\|\\+\\*\\?$"
        })""",
        R"""(
            root ::= "\"" ("[]{}()|+*?") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp quote",
        R"""({
            "type": "string",
            "pattern": "^\"$"
        })""",
        R"""(
            root ::= "\"" ("\"") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp with top-level alternation",
        R"""({
            "type": "string",
            "pattern": "^A|B|C|D$"
        })""",
        R"""(
            root ::= "\"" ("A" | "B" | "C" | "D") "\"" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "regexp",
        R"""({
            "type": "string",
            "pattern": "^(\\([0-9]{1,3}\\))?[0-9]{3}-[0-9]{4} a{3,5}nd...$"
        })""",
        R"""(
            dot ::= [^\x0A\x0D]
            root ::= "\"" (("(" root-1{1,3} ")")? root-1{3,3} "-" root-1{4,4} " " "a"{3,5} "nd" dot dot dot) "\"" space
            root-1 ::= [0-9]
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required props in original order",
        R"""({
            "type": "object",
            "properties": {
                "b": {"type": "string"},
                "c": {"type": "string"},
                "a": {"type": "string"}
            },
            "required": [
                "a",
                "b",
                "c"
            ],
            "additionalProperties": false,
            "definitions": {}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space b-kv "," space c-kv "," space a-kv "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "1 optional prop",
        R"""({
            "properties": {
                "a": {
                "type": "string"
                }
            },
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "N optional props",
        R"""({
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            a-rest ::= ( "," space b-kv )? b-rest
            b-kv ::= "\"b\"" space ":" space string
            b-rest ::= ( "," space c-kv )?
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            root ::= "{" space  (a-kv a-rest | b-kv b-rest | c-kv )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "required + optional props each in original order",
        R"""({
            "properties": {
                "b": {"type": "string"},
                "a": {"type": "string"},
                "d": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"],
            "additionalProperties": false
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space string
            b-kv ::= "\"b\"" space ":" space string
            c-kv ::= "\"c\"" space ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            d-kv ::= "\"d\"" space ":" space string
            d-rest ::= ( "," space c-kv )?
            root ::= "{" space b-kv "," space a-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "additional props",
        R"""({
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "number"}}
        })""",
        R"""(
            additional-kv ::= string ":" space additional-value
            additional-value ::= "[" space (number ("," space number)*)? "]" space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space  (additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "additional props (true)",
        R"""({
            "type": "object",
            "additionalProperties": true
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "additional props (implicit)",
        R"""({
            "type": "object"
        })""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= object
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "empty w/o additional props",
        R"""({
            "type": "object",
            "additionalProperties": false
        })""",
        R"""(
            root ::= "{" space  "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required + additional props",
        R"""({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "required": ["a"],
            "additionalProperties": {"type": "string"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space string
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space a-kv ( "," space ( additional-kv ( "," space additional-kv )* ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "optional + additional props",
        R"""({
            "type": "object",
            "properties": {
                "a": {"type": "number"}
            },
            "additionalProperties": {"type": "number"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space  (a-kv a-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "required + optional + additional props",
        R"""({
            "type": "object",
            "properties": {
                "and": {"type": "number"},
                "also": {"type": "number"}
            },
            "required": ["and"],
            "additionalProperties": {"type": "number"}
        })""",
        R"""(
            additional-k ::= ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space number
            also-kv ::= "\"also\"" space ":" space number
            also-rest ::= ( "," space additional-kv )*
            and-kv ::= "\"and\"" space ":" space number
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space and-kv ( "," space ( also-kv also-rest | additional-kv ( "," space additional-kv )* ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with empty name",
        R"""({
            "properties": {
                "": {"type": "integer"},
                "a": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            -kv ::= "\"\"" space ":" space root
            -rest ::= ( "," space a-kv )? a-rest
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] char+ | [^"a] char* ) ["] space
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= ("-"? integral-part) space
            root0 ::= "{" space  (-kv -rest | a-kv a-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with nested names",
        R"""({
            "properties": {
                "a": {"type": "integer"},
                "aa": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space integer
            a-rest ::= ( "," space aa-kv )? aa-rest
            aa-kv ::= "\"aa\"" space ":" space integer
            aa-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([a] char+ | [^"a] char*) | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (a-kv a-rest | aa-kv aa-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "optional props with common prefix",
        R"""({
            "properties": {
                "ab": {"type": "integer"},
                "ac": {"type": "integer"}
            },
            "additionalProperties": {"type": "integer"}
        })""",
        R"""(
            ab-kv ::= "\"ab\"" space ":" space integer
            ab-rest ::= ( "," space ac-kv )? ac-rest
            ac-kv ::= "\"ac\"" space ":" space integer
            ac-rest ::= ( "," space additional-kv )*
            additional-k ::= ["] ( [a] ([b] char+ | [c] char+ | [^"bc] char*) | [^"a] char* )? ["] space
            additional-kv ::= additional-k ":" space integer
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            integer ::= ("-"? integral-part) space
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            root ::= "{" space  (ab-kv ab-rest | ac-kv ac-rest | additional-kv ( "," space additional-kv )* )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "top-level $ref",
        R"""({
            "$ref": "#/definitions/foo",
            "definitions": {
                "foo": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "a"
                    ],
                    "additionalProperties": false
                }
            }
        })""",
        R"""(
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            ref-definitions-foo ::= "{" space ref-definitions-foo-a-kv "}" space
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space string
            root ::= ref-definitions-foo
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "anyOf",
        R"""({
            "anyOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                }
            },
            "type": "object"
        })""",
        R"""(
            alternative-0 ::= ref-definitions-foo
            alternative-1 ::= ref-definitions-bar
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            ref-definitions-bar ::= "{" space  (ref-definitions-bar-b-kv )? "}" space
            ref-definitions-bar-b-kv ::= "\"b\"" space ":" space number
            ref-definitions-foo ::= "{" space  (ref-definitions-foo-a-kv )? "}" space
            ref-definitions-foo-a-kv ::= "\"a\"" space ":" space number
            root ::= alternative-0 | alternative-1
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "anyOf $ref",
        R"""({
            "properties": {
                "a": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"}
                    ]
                },
                "b": {
                    "anyOf": [
                        {"$ref": "#/properties/a/anyOf/0"},
                        {"type": "boolean"}
                    ]
                }
            },
            "type": "object"
        })""",
        R"""(
            a ::= string | number
            a-kv ::= "\"a\"" space ":" space a
            a-rest ::= ( "," space b-kv )?
            b ::= b-0 | boolean
            b-0 ::= string
            b-kv ::= "\"b\"" space ":" space b
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space  (a-kv a-rest | b-kv )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
        )"""
    });

    test({
        SUCCESS,
        "mix of allOf, anyOf and $ref (similar to https://json.schemastore.org/tsconfig.json)",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"},
                {
                "anyOf": [
                    {"$ref": "#/definitions/baz"},
                    {"$ref": "#/definitions/bam"}
                ]
                }
            ],
            "definitions": {
                "foo": {
                    "properties": {"a": {"type": "number"}}
                },
                "bar": {
                    "properties": {"b": {"type": "number"}}
                },
                "bam": {
                    "properties": {"c": {"type": "number"}}
                },
                "baz": {
                    "properties": {"d": {"type": "number"}}
                }
            },
            "type": "object"
        })""",
        R"""(
            a-kv ::= "\"a\"" space ":" space number
            b-kv ::= "\"b\"" space ":" space number
            c-kv ::= "\"c\"" space ":" space number
            d-kv ::= "\"d\"" space ":" space number
            d-rest ::= ( "," space c-kv )?
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            root ::= "{" space a-kv "," space b-kv ( "," space ( d-kv d-rest | c-kv ) )? "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "allOf with enum schema",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b"]
                }
            }
        })""",
        R"""(
            root ::= ("\"a\"" | "\"b\"") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "allOf with multiple enum schemas",
        R"""({
            "allOf": [
                {"$ref": "#/definitions/foo"},
                {"$ref": "#/definitions/bar"}
            ],
            "definitions": {
                "foo": {
                    "type": "string",
                    "enum": ["a", "b", "c"]
                },
                "bar": {
                    "type": "string",
                    "enum": ["b", "c", "d"]
                }
            }
        })""",
        R"""(
            root ::= ("\"b\"" | "\"c\"") space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "conflicting names",
        R"""({
            "type": "object",
            "properties": {
                "number": {
                "type": "object",
                "properties": {
                    "number": {
                    "type": "object",
                        "properties": {
                            "root": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "root"
                        ],
                        "additionalProperties": false
                    }
                },
                "required": [
                    "number"
                ],
                "additionalProperties": false
                }
            },
            "required": [
                "number"
            ],
            "additionalProperties": false,
            "definitions": {}
        })""",
        R"""(
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            number- ::= "{" space number-number-kv "}" space
            number-kv ::= "\"number\"" space ":" space number-
            number-number ::= "{" space number-number-root-kv "}" space
            number-number-kv ::= "\"number\"" space ":" space number-number
            number-number-root-kv ::= "\"root\"" space ":" space number
            root ::= "{" space number-kv "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });

    test({
        SUCCESS,
        "description only (no type) treated as unconstrained",
        R"""({"description": "The 0-based index of the last line to be retrieved (inclusive). If None, read until the end of the file."})""",
        R"""(
            array ::= "[" space ( value ("," space value)* )? "]" space
            boolean ::= ("true" | "false") space
            char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
            decimal-part ::= [0-9]{1,16}
            integral-part ::= [0] | [1-9] [0-9]{0,15}
            null ::= "null" space
            number ::= ("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space
            object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? "}" space
            root ::= value
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
            string ::= "\"" char* "\"" space
            value ::= object | array | string | number | boolean | null
        )"""
    });

    test({
        SUCCESS,
        "literal string with escapes",
        R"""({
            "properties": {
                "code": {
                    "const": " \r \n \" \\ ",
                    "description": "Generated code",
                    "title": "Code",
                    "type": "string"
                }
            },
            "required": [
                "code"
            ],
            "title": "DecoderResponse",
            "type": "object"
        })""",
        R"""(
            code ::= "\" \\r \\n \\\" \\\\ \"" space
            code-kv ::= "\"code\"" space ":" space code
            root ::= "{" space code-kv "}" space
            space ::= | " " | "\n"{1,2} [ \t]{0,20}
        )"""
    });
}

static void test_resolves_to_string() {
    fprintf(stderr, "#\n# Testing resolves_to_string\n#\n");

    auto test = [](const std::string & name, const std::string & schema_str, bool expected) {
        fprintf(stderr, "- %s\n", name.c_str());
        common_schema_info info;
        auto schema = nlohmann::ordered_json::parse(schema_str);
        info.resolve_refs(schema);
        bool result = info.resolves_to_string(schema);
        if (result != expected) {
            fprintf(stderr, "#\n# Test '%s' failed.\n#\n", name.c_str());
            fprintf(stderr, "Schema: %s\n", schema_str.c_str());
            fprintf(stderr, "Expected: %s, Got: %s\n", expected ? "true" : "false", result ? "true" : "false");
            assert(false);
        }
    };

    // Basic type checks
    test("type string", R"({"type": "string"})", true);
    test("type integer", R"({"type": "integer"})", false);
    test("type number", R"({"type": "number"})", false);
    test("type boolean", R"({"type": "boolean"})", false);
    test("type object", R"({"type": "object"})", false);
    test("type array", R"({"type": "array"})", false);

    // Type array (nullable string)
    test("type array with string", R"({"type": ["string", "null"]})", true);
    test("type array without string", R"({"type": ["integer", "null"]})", false);

    // String-specific keywords
    test("minLength implies string", R"({"minLength": 1})", true);
    test("maxLength implies string", R"({"maxLength": 10})", true);
    test("pattern implies string", R"({"pattern": "^[a-z]+$"})", true);

    // Format
    test("format date", R"({"format": "date"})", true);
    test("format uuid", R"({"format": "uuid"})", true);
    test("format email", R"({"format": "email"})", true);

    // Const
    test("const string", R"({"const": "hello"})", true);
    test("const number", R"({"const": 123})", false);

    // Enum
    test("enum with strings", R"({"enum": ["a", "b", "c"]})", true);
    test("enum with numbers", R"({"enum": [1, 2, 3]})", false);
    test("enum mixed with string", R"({"enum": [1, "a", null]})", true);

    // anyOf
    test("anyOf with string", R"({"anyOf": [{"type": "string"}, {"type": "integer"}]})", true);
    test("anyOf without string", R"({"anyOf": [{"type": "integer"}, {"type": "boolean"}]})", false);

    // oneOf
    test("oneOf with string", R"({"oneOf": [{"type": "string"}, {"type": "number"}]})", true);
    test("oneOf without string", R"({"oneOf": [{"type": "object"}, {"type": "array"}]})", false);

    // allOf - all must be strings
    test("allOf all strings", R"({"allOf": [{"type": "string"}, {"minLength": 1}]})", true);
    test("allOf mixed types", R"({"allOf": [{"type": "string"}, {"type": "integer"}]})", false);

    // $ref
    test("$ref to string",
        R"({"$ref": "#/$defs/str", "$defs": {"str": {"type": "string"}}})", true);
    test("$ref to integer",
        R"({"$ref": "#/$defs/num", "$defs": {"num": {"type": "integer"}}})", false);

    // Nested
    test("nested anyOf with string",
        R"({"anyOf": [{"anyOf": [{"type": "integer"}, {"type": "string"}]}, {"type": "boolean"}]})", true);

    fprintf(stderr, "All resolves_to_string tests passed!\n");
}

int main() {
    fprintf(stderr, "LLAMA_NODE_AVAILABLE = %s\n", getenv("LLAMA_NODE_AVAILABLE") ? "true" : "false");
    fprintf(stderr, "LLAMA_PYTHON_AVAILABLE = %s\n", getenv("LLAMA_PYTHON_AVAILABLE") ? "true" : "false");

    test_resolves_to_string();

    test_all("C++", [](const TestCase & tc) {
        try {
            tc.verify(json_schema_to_grammar(nlohmann::ordered_json::parse(tc.schema), true));
            tc.verify_status(SUCCESS);
        } catch (const std::invalid_argument & ex) {
            fprintf(stderr, "Error: %s\n", ex.what());
            tc.verify_status(FAILURE);
        }
    });

    // C++ only tests (features not yet supported in JS/Python implementations)
    {
        fprintf(stderr, "#\n# Testing C++ only features\n#\n");
        auto run = [](const TestCase & tc) {
            fprintf(stderr, "- %s\n", tc.name.c_str());
            try {
                tc.verify(json_schema_to_grammar(nlohmann::ordered_json::parse(tc.schema), true));
                tc.verify_status(SUCCESS);
            } catch (const std::invalid_argument & ex) {
                fprintf(stderr, "Error: %s\n", ex.what());
                tc.verify_status(FAILURE);
            }
        };
        // Same as `run`, but for SUCCESS cases also feeds the generated grammar through
        // src/llama-grammar.cpp's own parser (llama_grammar_parser::parse) to confirm it is
        // valid, well-formed GBNF -- not just that the converter didn't throw. This matters
        // in particular for the PCRE shorthand character-class (\d, \w, \s, ...) translation
        // below, since the whole point of that fix is that llama-grammar.cpp's parser used to
        // reject the untranslated escapes at grammar-parse time.
        auto run_and_check_gbnf = [&](const TestCase & tc) {
            run(tc);
            if (tc.expected_status == SUCCESS) {
                tc.verify_expectation_parseable();
            }
        };

        run_and_check_gbnf({
            SUCCESS,
            "regexp with non-capturing group",
            R"""({
                "type": "string",
                "pattern": "^(?:foo|bar)baz$"
            })""",
            R"""(
                root ::= "\"" (("foo" | "bar") "baz") "\"" space
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            )""",
        });

        run_and_check_gbnf({
            SUCCESS,
            "regexp with nested non-capturing groups",
            R"""({
                "type": "string",
                "pattern": "^(?:(?:ab)+c)?d$"
            })""",
            R"""(
                root ::= "\"" ((("ab")+ "c")? "d") "\"" space
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            )""",
        });

        // Regression coverage for the PCRE shorthand character-class escapes (\d, \D, \w, \W,
        // \s, \S) that JSON Schema `pattern` regexes commonly use but that GBNF has no native
        // escape for: src/llama-grammar.cpp's parse_char() throws "unknown escape" if one reaches
        // it untranslated. A single such pattern anywhere in a combined tool-calling grammar used
        // to disable grammar-constrained decoding for the whole request (confirmed against a
        // PagerDuty `create_schedule` MCP tool schema using a leap-year-validated ISO-8601
        // `pattern`, e.g. containing "\d\d[2468][048]", that reliably logged
        // "parse: error parsing grammar: unknown escape at \d\d...").

        run_and_check_gbnf({
            SUCCESS,
            "regexp with \\d \\w \\s shorthand classes inside [...] (mixed with literal members)",
            R"""({
                "type": "string",
                "pattern": "^[\\dA-F]{4}-[\\w.-]+ [\\s,;]$"
            })""",
            R"""(
                root ::= "\"" (root-1{4,4} "-" [A-Za-z0-9_.-]+ " " [ \t\n\r,;]) "\"" space
                root-1 ::= [0-9A-F]
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            )""",
        });

        run_and_check_gbnf({
            SUCCESS,
            "regexp with standalone \\d \\w \\s shorthand classes and quantifiers",
            R"""({
                "type": "string",
                "pattern": "^\\d+\\.\\w+\\s?$"
            })""",
            R"""(
                d ::= [0-9]
                root ::= "\"" (d+ "." w+ s?) "\"" space
                s ::= [ \t\n\r]
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
                w ::= [A-Za-z0-9_]
            )""",
        });

        run_and_check_gbnf({
            SUCCESS,
            "regexp with standalone negated shorthand classes \\D \\W \\S",
            R"""({
                "type": "string",
                "pattern": "^\\D\\W\\S$"
            })""",
            R"""(
                not-d ::= [^0-9]
                not-s ::= [^ \t\n\r]
                not-w ::= [^A-Za-z0-9_]
                root ::= "\"" (not-d not-w not-s) "\"" space
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            )""",
        });

        run_and_check_gbnf({
            SUCCESS,
            "regexp shaped like an ISO-8601 date (\\d immediately after literal '-', "
            "exercising \\d recognition mid-literal-run, not just at the start of a token)",
            R"""({
                "type": "string",
                "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
            })""",
            R"""(
                d ::= [0-9]
                root ::= "\"" (root-1{4,4} "-" root-1{2,2} "-" root-1{2,2}) "\"" space
                root-1 ::= d
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            )""",
        });

        run_and_check_gbnf({
            SUCCESS,
            "regexp shaped like the PagerDuty leap-year-validated ISO-8601 pattern that broke "
            "grammar-constrained decoding in production (\\d\\d[2468][048] etc, top-level "
            "alternation)",
            R"""({
                "type": "string",
                "pattern": "^\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]$"
            })""",
            R"""(
                d ::= [0-9]
                root ::= "\"" (d d [2468] [048] | d d [13579] [26] | d d "0" [48]) "\"" space
                space ::= | " " | "\n"{1,2} [ \t]{0,20}
            )""",
        });

        // \D, \W, \S mixed inside a [...] class alongside other members have no clean
        // single-range GBNF translation (a positive class can't compose with a negated-shorthand
        // member), so conversion must fail loudly and *name the offending pattern* right here at
        // schema-conversion time, rather than emitting grammar text that only fails later --
        // without any schema context -- inside llama-grammar.cpp's parser.
        run({
            FAILURE,
            "regexp with \\D negated shorthand mixed inside [...] must fail at conversion time, not silently produce invalid GBNF",
            R"""({
                "type": "string",
                "pattern": "^[\\D!?]+$"
            })""",
            ""
        });

        // Verify the failure above is a clean, actionable error (names the pattern and the
        // offending escape), not a generic/opaque failure -- this is the property that makes it
        // debuggable at conversion time instead of a bare "unknown escape" from the GBNF parser
        // with no indication of which schema or pattern caused it.
        {
            const std::string bad_pattern = "^[\\D!?]+$";
            bool threw = false;
            try {
                json_schema_to_grammar(nlohmann::ordered_json::parse(
                    R"({"type": "string", "pattern": "^[\\D!?]+$"})"), true);
            } catch (const std::invalid_argument & ex) {
                threw = true;
                std::string msg = ex.what();
                fprintf(stderr, "- negated shorthand class error message: %s\n", msg.c_str());
                if (msg.find(bad_pattern) == std::string::npos || msg.find("\\D") == std::string::npos) {
                    fprintf(stderr, "# FAILED: error message does not name the offending pattern/escape:\n%s\n", msg.c_str());
                    assert(false);
                }
            }
            if (!threw) {
                fprintf(stderr, "# FAILED: expected a negated shorthand class mixed inside [...] to raise std::invalid_argument\n");
                assert(false);
            }
        }
    }

    if (getenv("LLAMA_SKIP_TESTS_SLOW_ON_EMULATOR")) {
        fprintf(stderr, "\033[33mWARNING: Skipping slow tests on emulator.\n\033[0m");
    } else {
        if (getenv("LLAMA_PYTHON_AVAILABLE") || (std::system("python -c \"import sys; exit(1) if sys.version_info < (3, 8) else print('Python version is sufficient')\"") == 0)) {
            test_all("Python", [](const TestCase & tc) {
                write("test-json-schema-input.tmp", tc.schema);
                tc.verify_status(std::system(
                    "python ./examples/json_schema_to_grammar.py test-json-schema-input.tmp > test-grammar-output.tmp") == 0 ? SUCCESS : FAILURE);
                tc.verify(read("test-grammar-output.tmp"));
            });
        } else {
            fprintf(stderr, "\033[33mWARNING: Python not found (min version required is 3.8), skipping Python JSON schema -> grammar tests.\n\033[0m");
        }
    }

    test_all("Check Expectations Validity", [](const TestCase & tc) {
        if (tc.expected_status == SUCCESS) {
            tc.verify_expectation_parseable();
        }
    });
}
