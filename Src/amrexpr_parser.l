%option noyywrap nodefault
%option never-interactive
%{
#include "amrexpr_Parser_Y.H"
#include "amrexpr_parser.tab.h"
#ifdef _WIN32
#define YY_NO_UNISTD_H
#endif
%}

%option nounput
%option noinput
%option prefix="amrexpr_parser"

 /* Tokens NUMBER, SYMBOL, F1, POW, F2, etc. are defined in amrexpr_parser.y. */
 /* Types PARSER_SQRT, PARSER_EXP, etc. are defined in amrexpr_Parser_y.H. */

 /* Used later to define NUMBER */
EXP	([Ee][-+]?[0-9]+)

%%

"+" |
"-" |
"*" |
"/" |
"=" |
"," |
"<" |
">" |
";" |
"(" |
")"     { return amrexpr_parsertext[0]; } /* simply pass through */

 /* amrexpr_parserlval is union type defined in amrexpr_parser.tab.h that is generated
  * by bison with amrexpr_parser.y */

"sqrt"        { amrexpr_parserlval.f1 = amrexpr::PARSER_SQRT;      return F1; }
"exp"         { amrexpr_parserlval.f1 = amrexpr::PARSER_EXP;       return F1; }
"log"         { amrexpr_parserlval.f1 = amrexpr::PARSER_LOG;       return F1; }
"log10"       { amrexpr_parserlval.f1 = amrexpr::PARSER_LOG10;     return F1; }
"sin"         { amrexpr_parserlval.f1 = amrexpr::PARSER_SIN;       return F1; }
"cos"         { amrexpr_parserlval.f1 = amrexpr::PARSER_COS;       return F1; }
"tan"         { amrexpr_parserlval.f1 = amrexpr::PARSER_TAN;       return F1; }
"asin"        { amrexpr_parserlval.f1 = amrexpr::PARSER_ASIN;      return F1; }
"acos"        { amrexpr_parserlval.f1 = amrexpr::PARSER_ACOS;      return F1; }
"atan"        { amrexpr_parserlval.f1 = amrexpr::PARSER_ATAN;      return F1; }
"sinh"        { amrexpr_parserlval.f1 = amrexpr::PARSER_SINH;      return F1; }
"cosh"        { amrexpr_parserlval.f1 = amrexpr::PARSER_COSH;      return F1; }
"tanh"        { amrexpr_parserlval.f1 = amrexpr::PARSER_TANH;      return F1; }
"asinh"       { amrexpr_parserlval.f1 = amrexpr::PARSER_ASINH;     return F1; }
"acosh"       { amrexpr_parserlval.f1 = amrexpr::PARSER_ACOSH;     return F1; }
"atanh"       { amrexpr_parserlval.f1 = amrexpr::PARSER_ATANH;     return F1; }
"abs"         { amrexpr_parserlval.f1 = amrexpr::PARSER_ABS;       return F1; }
"fabs"        { amrexpr_parserlval.f1 = amrexpr::PARSER_ABS;       return F1; }
"floor"       { amrexpr_parserlval.f1 = amrexpr::PARSER_FLOOR;     return F1; }
"ceil"        { amrexpr_parserlval.f1 = amrexpr::PARSER_CEIL;      return F1; }
"comp_ellint_1" { amrexpr_parserlval.f1 = amrexpr::PARSER_COMP_ELLINT_1; return F1; }
"comp_ellint_2" { amrexpr_parserlval.f1 = amrexpr::PARSER_COMP_ELLINT_2; return F1; }
"erf"         { amrexpr_parserlval.f1 = amrexpr::PARSER_ERF;       return F1; }
"**"          { amrexpr_parserlval.f2 = amrexpr::PARSER_POW;       return POW;}
"^"           { amrexpr_parserlval.f2 = amrexpr::PARSER_POW;       return POW;}
"atan2"       { amrexpr_parserlval.f2 = amrexpr::PARSER_ATAN2;     return F2; }
">="          { amrexpr_parserlval.f2 = amrexpr::PARSER_GEQ;       return GEQ;}
"<="          { amrexpr_parserlval.f2 = amrexpr::PARSER_LEQ;       return LEQ;}
"=="          { amrexpr_parserlval.f2 = amrexpr::PARSER_EQ;        return EQ;}
"!="          { amrexpr_parserlval.f2 = amrexpr::PARSER_NEQ;       return NEQ;}
"and"         { amrexpr_parserlval.f2 = amrexpr::PARSER_AND;       return AND;}
"or"          { amrexpr_parserlval.f2 = amrexpr::PARSER_OR;        return OR;}
"pow"         { amrexpr_parserlval.f2 = amrexpr::PARSER_POW;       return F2; }
"heaviside"   { amrexpr_parserlval.f2 = amrexpr::PARSER_HEAVISIDE; return F2; }
"jn"          { amrexpr_parserlval.f2 = amrexpr::PARSER_JN;        return F2; }
"min"         { amrexpr_parserlval.f2 = amrexpr::PARSER_MIN;       return F2; }
"max"         { amrexpr_parserlval.f2 = amrexpr::PARSER_MAX;       return F2; }
"fmod"        { amrexpr_parserlval.f2 = amrexpr::PARSER_FMOD;      return F2; }
"if"          { amrexpr_parserlval.f3 = amrexpr::PARSER_IF;        return F3; }

 /* We use SYMBOL to hold variables and constants */
[a-zA-Z_][a-zA-Z0-9_\.]*  { amrexpr_parserlval.s = amrexpr::parser_makesymbol(amrexpr_parsertext); return SYMBOL; }

 /* Number */
[0-9]+"."[0-9]*{EXP}? |
"."?[0-9]+{EXP}?  { amrexpr_parserlval.d = std::atof(amrexpr_parsertext);   return NUMBER; }

 /* Special characters */
"//".*
[ \t]   /* ignore white space */
\\\n    /* ignore line continuation */
"\n"    { return EOL; }

 /* everything else */
.	{ amrexpr_parsererror("Unknown character %c", *amrexpr_parsertext); }

%%
