use strict;
use warnings;

#extract dictionary from alignment probabilities
#Etract pairs with maximum alignment probability
#  only if the max is unique


die "Num of args ".scalar(@ARGV) . "Usage: perl extract_max_dictionary.pl alignment_p_file output_file \n" if (@ARGV < 2);
my $input_file = $ARGV[0];
my $output_file = $ARGV[1];
open(O1, ">$output_file") || die "Cannot create $output_file\n";
open(I, $input_file) || die "Cannot open $input_file\n";

my $count=0;
my $max=-99999;
my $max_tr="";
my $current_word="";
while (my $str = <I>) {
    $str =~ s/\s+$//;
    my @tokens = split(/\s+/, $str);
    my $w1 = $tokens[0];
    my $tr = $tokens[1];
    my $p = $tokens[2];
    if ($w1 ne $current_word){
        if ($count == 1 and $current_word ne '<eps>'){
           print O1 "$current_word\t$max_tr\n";
        }
        $current_word=$w1;
        $max_tr="";
        $max=-99999;
        $count=0;
     }
     if ($p > $max){
        $max=$p;
        $max_tr=$tr;
        $count=1;
     }elsif ($p eq $max){
        $count++;
     }
}

if ($count == 1){
  print O1 "$current_word\t$max_tr\n";
}

close(I);
close(O1);
