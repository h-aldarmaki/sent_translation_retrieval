use strict;
use warnings;

die "Num of args ".scalar(@ARGV) . "Usage: perl calculate_sif.pl file sif_file\n" if (@ARGV < 2);
my $input_file = $ARGV[0];
my $sif_file = $ARGV[1];

my $N=0;
my %freq;
open(I, $input_file) || die "Cannot open $input_file\n";
while (my $str = <I>) {
    $str =~ s/\s+$//;
    my @words = split(/\s+/, $str);
    foreach my $w (@words) {
      $freq{$w}++;
      $N++;
    }
}
close(I);

my $a=0.001;
foreach my $term (keys %freq) {
    $freq{$term}=$a/($a+($freq{$term}/$N));
}
    
open(O, ">$sif_file") || die "Cannot create $sif_file\n";
foreach my $term (keys %freq) { #(sort {$a <=> $b} keys %$idf) {
      print O "$term $freq{$term}\n";
}
close(O);
