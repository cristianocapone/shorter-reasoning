"""The MATH checker of synth/math_harvest.py: unchanged on string answers (differential test against a verbatim copy of the
2026-10-25 implementation), integer answers (AIME), and the answer_first_char rule behind ans_tok. No tokenizer needed."""
import random, re, unittest

import test_eval_util  # noqa: F401  (puts synth/ on sys.path)
import math_harvest as M


# ---- verbatim copy of the original checker (synth/math_harvest.py as of 2026-10-25) --------------------------------
def _orig_boxed(text):
    for part in ([text.split("</think>")[-1]] if "</think>" in text else []) + [text]:
        i = part.rfind("\\boxed")
        if i < 0:
            continue
        j = part.find("{", i)
        if j < 0:
            continue
        d = 0
        for k in range(j, len(part)):
            d += part[k] == "{"; d -= part[k] == "}"
            if d == 0:
                return part[j + 1:k]
    return None


def _orig_norm(s):
    if s is None:
        return None
    s = s.strip()
    for x in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", " ", "\\displaystyle", "^\\circ", "^{\\circ}", "\\%", "%"):
        s = s.replace(x, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("\\text{", "{").replace("\\mbox{", "{")
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"(\\frac\{[^}]*\})(\d)", r"\1{\2}", s)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]
    try:
        f = float(s.replace(",", ""))
        return f"{f:.6g}"
    except ValueError:
        return s


def _orig_correct(text, ref):
    p = _orig_norm(_orig_boxed(text)); r = _orig_norm(ref)
    return p is not None and p == r
# ---------------------------------------------------------------------------------------------------------------------

ANSWERS = ["5", " 5 ", "5.0", "5.", "-3", "0", "025", "1,000", "1000", "1234567", "0.333333333", "\\frac{3}{4}", "\\dfrac{3}{4}",
           "\\tfrac34", "\\frac34", "\\frac{1}2", "\\frac{\\sqrt{3}}{2}", "2\\sqrt{3}", "\\sqrt{2}", "\\pi", "2\\pi", "\\infty",
           "(-\\infty, 3]", "\\left( 3, \\frac{\\pi}{2} \\right)", "[1,2)", "x^2+2x+1", "x^2 + 2x + 1", "30^\\circ",
           "30^{\\circ}", "10\\%", "10%", "\\$25", "\\text{(C)}", "(C)", "\\textbf{(A)}", "{12}", "\\{1,2\\}", "", "   ",
           "\\mbox{Monday}", "\\text{Monday}", "y = 2x + 3", "\\displaystyle\\frac{1}{2}", "3\\!\\cdot\\!4", "1\\,000",
           "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "a,b", "\\frac{1}{2}.", "6.022e23", "1e6", "-\\frac{1}{2}"]


class TestBackwardCompatibility(unittest.TestCase):
    def test_norm_identical_on_strings(self):
        for s in ANSWERS + [None]:
            self.assertEqual(M.norm(s), _orig_norm(s), s)

    def test_norm_identical_on_random_strings(self):
        rng = random.Random(0)
        alphabet = ["1", "2", "0", "5", ".", ",", "-", " ", "{", "}", "\\frac", "\\dfrac", "\\left", "\\right", "$", "x", "^",
                    "\\circ", "%", "\\%", "\\text{", "(", ")", "\\sqrt", "\\pi", "\\!", "e", "+", "/"]
        for _ in range(5000):
            s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 9)))
            self.assertEqual(M.norm(s), _orig_norm(s), repr(s))

    def test_correct_and_boxed_identical(self):
        texts = ["<think>\nso 5\n</think>\n\nThe answer is $\\boxed{5}$.", "no box here", "\\boxed{3} then \\boxed{\\frac{3}{4}}",
                 "<think>\\boxed{7}\n</think>\n\nanswer \\boxed{8}", "<think>cut in the middle \\boxed{9}", "\\boxed{{1,2}}",
                 "\\boxed{\\text{(C)}}", "\\boxed 5", "<think>x</think>\n\nno box after think, \\boxed{4} before? no"]
        for t in texts:
            self.assertEqual(M.boxed(t), _orig_boxed(t), t)
            for ref in ANSWERS:
                self.assertEqual(M.correct(t, ref), _orig_correct(t, ref), (t, ref))


class TestIntegerAnswers(unittest.TestCase):
    def test_int_reference(self):
        self.assertTrue(M.correct("<think>\n...\n</think>\n\nSo the answer is $\\boxed{204}$.", 204))
        self.assertTrue(M.correct("\\boxed{025}", 25))          # AIME answers are often written with leading zeros
        self.assertTrue(M.correct("\\boxed{25}", "025"))
        self.assertTrue(M.correct("\\boxed{0}", 0))
        self.assertTrue(M.correct("\\boxed{204.}", 204))
        self.assertTrue(M.correct("\\boxed{\\text{204}}", 204))
        self.assertFalse(M.correct("\\boxed{26}", 25))
        self.assertFalse(M.correct("the answer is 25", 25))      # no \boxed{}: not graded
        self.assertFalse(M.correct("\\boxed{m+n=25}", 25))       # unchanged rule: exact normalised match only
        self.assertEqual(M.norm(204), M.norm("204"))
        self.assertEqual(M.norm(7), "7")

    def test_original_checker_crashed_on_int(self):
        with self.assertRaises(AttributeError):              # why the int coercion was needed
            _orig_correct("\\boxed{25}", 25)


class TestAnswerFirstChar(unittest.TestCase):
    def first(self, text, ans):
        return M.answer_first_char(text, ans)

    def test_none(self):
        self.assertIsNone(self.first("anything", None))

    def test_first_plain_mention(self):
        t = "<think>\nWe get x = 12. Check: 3*4 = 12.\n</think>\n\n\\boxed{12}"
        self.assertEqual(self.first(t, "12"), t.index("x = 12") + 4)

    def test_boxed_only(self):
        t = "<think>\nlong reasoning without the number\n</think>\n\nThe answer is \\boxed{\\dfrac{3}{4}}."
        self.assertEqual(self.first(t, "\\dfrac{3}{4}"), t.index("\\boxed"))

    def test_earlier_boxed_equal_up_to_norm(self):
        t = "<think>so \\boxed{\\frac34} maybe</think> final \\boxed{\\dfrac{3}{4}}"
        self.assertEqual(self.first(t, "\\dfrac{3}{4}"), t.index("\\boxed"))

    def test_not_part_of_longer_numbers_or_expressions(self):
        for noise in ("15 apples", "1.5 m", "x-5 = 0", "5x", "5^2", "\\frac{5}{7}", "\\frac{7}{5}", "\\sqrt{5}", "x_5",
                      "50", "5,000", "5.5", "2/5", "5/2", "a5", "\\5"):
            t = f"we see {noise} here; finally x = 5.\n</think>\n\n\\boxed{{5}}"
            self.assertEqual(self.first(t, "5"), t.index("x = 5") + 4, noise)

    def test_allowed_contexts(self):
        for ctx, off in (("x = 5 apples", 4), ("($5$)", 2), ("5^\\circ", 0), ("5%", 0), ("is 5, then", 3), ("5.\n", 0)):
            t = f"intro {ctx} more\n</think>\n\n\\boxed{{5}}"
            self.assertEqual(self.first(t, "5"), t.index(ctx) + off, ctx)
        t = "so y = -5 here\n</think>\\boxed{-5}"                    # a negative answer may follow '='
        self.assertEqual(self.first(t, "-5"), t.index("-5"))

    def test_whitespace_insensitive(self):
        t = "the root is x^2 + 2x + 1 so\n</think>\\boxed{x^2+2x+1}"
        self.assertEqual(self.first(t, "x^2+2x+1"), t.index("x^2 + 2x"))

    def test_overlapping_occurrence(self):
        t = "pairs 1,1,1 and (1,1) end"
        self.assertEqual(self.first(t, "1,1"), t.index("1,1,1") + 2)    # the first standalone '1,1' overlaps the first match

    def test_integer_answer_and_leading_zero(self):
        t = "thus m+n = 25 \\boxed{025}"
        self.assertEqual(self.first(t, 25), t.index("25"))
        self.assertEqual(self.first("\\boxed{025}", 25), 0)

    def test_hit_guaranteed_when_boxed(self):
        rng = random.Random(1)
        for _ in range(300):
            ans = rng.choice(["7", "\\frac{1}{2}", "x+1", "(2,3)", "42", "\\sqrt{2}"])
            filler = "".join(rng.choice("abc 123+-=.,{}^_\\") for _ in range(200))
            t = filler + "\n</think>\n\n\\boxed{" + ans + "}"
            pos = self.first(t, M.boxed(t))
            self.assertIsNotNone(pos)
            self.assertLessEqual(pos, t.rindex("\\boxed"))


if __name__ == "__main__":
    unittest.main()
