
%{
#include "amrexpr_Parser_Y.H"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int amrexpr_parserlex (void);
/* Bison seems to have a bug. yyalloc etc. do not have the api.prefix. */
#ifndef yyalloc
#  define yyalloc amrexpr_parseralloc
#endif
#ifndef yysymbol_kind_t
#  define yysymbol_kind_t amrexpr_parsersymbol_kind_t
#endif
%}

/* We do not need to make this reentrant safe, because we use flex and
   bison for generating AST only and this part doesn't need to be
   thread safe.
*/
/*%define api.pure full */
%define api.prefix {amrexpr_parser}

/* This is the type returned by functions parser_new* declared in
   amrexpr_Parser_y.H.  See also bison rules at the end of this file.
*/
%union {
    struct amrexpr::parser_node* n;
    double d;
    struct amrexpr::parser_symbol* s;
    enum amrexpr::parser_f1_t f1;
    enum amrexpr::parser_f2_t f2;
    enum amrexpr::parser_f3_t f3;
}

/* Define tokens.  They are used by flex too. */
%token <n>  NODE
%token <d>  NUMBER
%token <s>  SYMBOL
%token <f1> F1
%token <f2> F2
%token <f3> F3
%token EOL
%token POW "**" '^'
%token GEQ ">="
%token LEQ "<="
%token EQ "=="
%token NEQ "!="
%token AND "and"
%token OR "or"

%left ';'
%nonassoc F1 F2 F3
%right '='
%left OR
%left AND
%left EQ NEQ
%left '<' '>' GEQ LEQ
%left '+' '-'
%left '*' '/'
%nonassoc NEG UPLUS
%right POW

/* This specifies the type of expressions */
%type <n> exp stmt or_exp and_exp cmp_exp add_exp mul_exp pow_exp unary_exp primary_exp

%start input

%%

/* Given `\n` terminated input, a tree is generated and passed to
 * function parser_defexpr defined in amrexpr_Parser_Y.cpp.
 */
input:
  %empty
| input exp EOL {
    amrexpr::parser_defexpr($2);
  }
;

/* Top level - handles lists and assignments */
exp:
  stmt                       { $$ = $1; }
| exp ';' stmt               { $$ = amrexpr::parser_newlist($1, $3); }
| exp ';'                    { $$ = amrexpr::parser_newlist($1, nullptr); }
;

/* Statements - handles assignments and expressions */
stmt:
  or_exp                     { $$ = $1; }
| SYMBOL '=' or_exp          { $$ = amrexpr::parser_newassign($1, $3); }

/* OR expressions */
or_exp:
  and_exp                    { $$ = $1; }
| or_exp OR and_exp          { $$ = amrexpr::parser_newf2(amrexpr::PARSER_OR, $1, $3); }
;

/* AND expressions */
and_exp:
  cmp_exp                    { $$ = $1; }
| and_exp AND cmp_exp        { $$ = amrexpr::parser_newf2(amrexpr::PARSER_AND, $1, $3); }
;

/* Comparison expressions - handles all comparison operators and chaining */
cmp_exp:
  add_exp                    { $$ = $1; }
| cmp_exp '<' add_exp        { $$ = amrexpr::parser_newcmpchain($1, amrexpr::PARSER_LT, $3); }
| cmp_exp '>' add_exp        { $$ = amrexpr::parser_newcmpchain($1, amrexpr::PARSER_GT, $3); }
| cmp_exp LEQ add_exp        { $$ = amrexpr::parser_newcmpchain($1, amrexpr::PARSER_LEQ,$3); }
| cmp_exp GEQ add_exp        { $$ = amrexpr::parser_newcmpchain($1, amrexpr::PARSER_GEQ,$3); }
| cmp_exp EQ add_exp         { $$ = amrexpr::parser_newcmpchain($1, amrexpr::PARSER_EQ ,$3); }
| cmp_exp NEQ add_exp        { $$ = amrexpr::parser_newcmpchain($1, amrexpr::PARSER_NEQ,$3); }
;

/* Addition and subtraction */
add_exp:
  mul_exp                    { $$ = $1; }
| add_exp '+' mul_exp        { $$ = amrexpr::parser_newnode(amrexpr::PARSER_ADD, $1, $3); }
| add_exp '-' mul_exp        { $$ = amrexpr::parser_newnode(amrexpr::PARSER_SUB, $1, $3); }
;

/* Multiplication and division */
mul_exp:
  unary_exp                  { $$ = $1; }
| mul_exp '*' unary_exp      { $$ = amrexpr::parser_newnode(amrexpr::PARSER_MUL, $1, $3); }
| mul_exp '/' unary_exp      { $$ = amrexpr::parser_newnode(amrexpr::PARSER_DIV, $1, $3); }
;

/* Unary expressions */
unary_exp:
  pow_exp                    { $$ = $1; }
| '-' unary_exp              { $$ = amrexpr::parser_newneg($2); }
| '+' unary_exp              { $$ = $2; }
;

/* Power (right associative) */
pow_exp:
  primary_exp                { $$ = $1; }
| primary_exp POW unary_exp  { $$ = amrexpr::parser_newf2(amrexpr::PARSER_POW, $1, $3); }
;

/* Primary expressions */
primary_exp:
  NUMBER                     { $$ = amrexpr::parser_newnumber($1); }
| SYMBOL                     { $$ = amrexpr::parser_newsymbol($1); }
| '(' or_exp ')'                { $$ = $2; }
| F1 '(' or_exp ')'             { $$ = amrexpr::parser_newf1($1, $3); }
| F2 '(' or_exp ',' or_exp ')'     { $$ = amrexpr::parser_newf2($1, $3, $5); }
| F3 '(' or_exp ',' or_exp ',' or_exp ')' { $$ = amrexpr::parser_newf3($1, $3, $5, $7); }
;

%%
