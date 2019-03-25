use strict;
use warnings;

#extract aligned word indices for source and target languages. 
#only 1-to-1 alignments will be extracted for each sentence. 
#Output will be space seaparted lists, tab separated for source and target

die "Num of args ".scalar(@ARGV) . "Usage: perl extract_aligned_pairs.pl alignment_file output_file other_lang \n" if (@ARGV < 3);
my $input_file = $ARGV[0];
my $output_file = $ARGV[1];
my $lang = $ARGV[2];
open(O1, ">$output_file.en") || die "Cannot create $output_file.en\n";
open(O2, ">$output_file.$lang") || die "Cannot create $output_file.$lang\n";
open(I, $input_file) || die "Cannot open $input_file\n";
my $max_len=500; #maximum length of sentence to consider for alignment
while (my $str = <I>) {
    $str =~ s/\s+$//;
    my @words = split(/\s+/, $str);
    my %a1;
    my %a2; #dump
    foreach my $w (@words) {
      my @t = split(/\-/, $w);
      if (not exists $a2{$t[1]}){ 
         if (exists $a1{$t[0]}){
              $a1{$t[0]}=-1;
         }else{
           $a1{$t[0]}=$t[1];
         }
         $a2{$t[1]}=$t[0];
      }else{ #if target word is already aligned
        $a1{$t[0]}=-1;
        $a1{$a2{$t[1]}}=-1;
      }
      
    }
    for (my $j=0; $j<@words; $j++){
        if ($j>=$max_len) { last;}
        if (exists $a1{$j} and $a1{$j} >-1 and $a1{$j} < $max_len){
          print O1 "$j "; 
          print O2 "$a1{$j} "
       
        }
    }
   print O1 "\n";
   print O2 "\n";
}
close(I);
close(O1);
close(O2);
